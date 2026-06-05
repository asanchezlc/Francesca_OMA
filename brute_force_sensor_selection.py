# -*- coding: utf-8 -*-
"""
Brute-force validation for 3-sensor selection.

This script evaluates ALL combinations of k sensors using the SAME metrics and
QUBO matrix used by the main qaoa-fm_OMA.py workflow, so the resulting ranking
can be compared directly against the QAOA solution.

Typical use:
    python brute_force_sensor_selection.py

Notes
-----
- The script imports the main project file by path, so the original code does
  not need to be duplicated.
- It does NOT run QAOA. It only performs exhaustive evaluation of all subsets.
- Ranking is based on the same QUBO energy used by the optimization workflow:
      E(x) = x^T Q x + lambda * k^2
  Since the cardinality is fixed to k for every tested subset, the ranking is
  fully consistent with the main objective.
"""

from __future__ import annotations

import itertools
import importlib.util
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "grid.linewidth": 0.6,
})

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
MAIN_QAOA_FILE = BASE_DIR / "qaoa-fm_OMA.py"

EXCEL_FILE = BASE_DIR / "Roberto_signal_Puente_Circunv.xlsx"
GEOMETRY_FILE = BASE_DIR / "signal_Geo.xlsx"
OMA_JSON_FILE = BASE_DIR / "results" / "EFDD_results.json"

SAMPLE_RATE_HZ = 100.0
K_SENSORS_TO_SELECT = 3

# Utility weights
W_ENTROPY = 0.3
W_SPECTRAL = 0.3
W_FIM = 0.4

# Redundancy weights
W_MI = 0.5
W_BAND_CORR = 0.5
N_SPECTRAL_BANDS = 4

# Geometry / symmetry / cardinality weights
SYMMETRY_COORD = "x"
GEOM_LENGTH_SCALE = 20.0
GEOM_GAMMA = 1.0
SYM_ETA = 1.0
QUBO_LAMBDA = 5.0
QUBO_ALPHA = 1.0

# Optional: compare with the QAOA solution already obtained
QAOA_SELECTED_SENSORS = None
# Example:
# QAOA_SELECTED_SENSORS = ["Sensor1", "Sensor2", "Sensor8"]

# Output files
OUT_DIR = BASE_DIR / "bruteforce_results"
OUT_DIR.mkdir(exist_ok=True)
RANKING_CSV = OUT_DIR / "bruteforce_ranking_all_combinations.csv"
TOP10_CSV = OUT_DIR / "bruteforce_top10.csv"
SUMMARY_TXT = OUT_DIR / "bruteforce_summary.txt"
RANKING_PNG = OUT_DIR / "bruteforce_ranking_all.png"
TOP10_PNG = OUT_DIR / "bruteforce_top10.png"
BREAKDOWN_TOP10_PNG = OUT_DIR / "bruteforce_top10_breakdown.png"
HEATMAP_TOP10_PNG = OUT_DIR / "bruteforce_top10_heatmap.png"


# =============================================================================
# Helpers
# =============================================================================
def load_main_module(pyfile: Path):
    """Load qaoa-fm_OMA.py by path, even though its filename contains a hyphen."""
    if not pyfile.exists():
        raise FileNotFoundError(f"Main QAOA file not found: {pyfile}")

    spec = importlib.util.spec_from_file_location("qaoa_main_module", pyfile)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {pyfile}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subset_energy_from_Q(Q: np.ndarray, selected_indices: List[int], constant_offset: float = 0.0) -> float:
    """Evaluate E(x) = x^T Q x + constant_offset for a fixed binary subset."""
    n = Q.shape[0]
    x = np.zeros(n, dtype=float)
    x[selected_indices] = 1.0
    return float(x @ Q @ x + constant_offset)


def decompose_subset_terms(
    selected_indices: List[int],
    b: np.ndarray,
    Corr: np.ndarray,
    prox_matrix: np.ndarray | None,
    left_indices: np.ndarray | None,
    right_indices: np.ndarray | None,
    alpha: float,
    gamma: float,
    eta_sym: float,
    lam: float,
    k: int,
) -> dict:
    """Return a transparent decomposition of the objective for one subset."""
    n = len(b)
    x = np.zeros(n, dtype=float)
    x[selected_indices] = 1.0

    utility_term = -float(np.sum(x * b))

    redundancy_term = 0.0
    geometry_term = 0.0
    for a, i in enumerate(selected_indices):
        for j in selected_indices[a + 1 :]:
            redundancy_term += alpha * float(Corr[i, j])
            if prox_matrix is not None:
                geometry_term += gamma * float(prox_matrix[i, j])

    symmetry_term = 0.0
    if left_indices is not None and right_indices is not None:
        left_count = float(np.sum(x[left_indices]))
        right_count = float(np.sum(x[right_indices]))
        symmetry_term = eta_sym * (left_count - right_count) ** 2
    else:
        left_count = np.nan
        right_count = np.nan

    cardinality_term = lam * (float(np.sum(x)) - k) ** 2

    total_energy = utility_term + redundancy_term + geometry_term + symmetry_term + cardinality_term

    return {
        "utility_term": utility_term,
        "redundancy_term": redundancy_term,
        "geometry_term": geometry_term,
        "symmetry_term": symmetry_term,
        "cardinality_term": cardinality_term,
        "total_energy_from_terms": total_energy,
        "left_count": left_count,
        "right_count": right_count,
    }

def plot_full_ranking(ranking: pd.DataFrame, qaoa_selected_sensors=None, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    ax.plot(
        ranking["rank"],
        ranking["energy"],
        color="#1f4e79",
        marker="o",
        markersize=3.0,
        linewidth=1.4,
        label="All subsets",
    )

    ax.set_title("Sorted objective values for all 3-sensor subsets", pad=10)
    ax.set_xlabel("Subset rank")
    ax.set_ylabel(r"Objective value $J(S)$")

    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_xlim(1, len(ranking))
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()

def plot_top10_ranking(ranking: pd.DataFrame, save_path: Path | None = None):
    top10 = ranking.head(10).copy()
    top10["subset_label"] = top10["subset_sensors"].apply(lambda s: s.replace("Sensor", "S"))

    # reverse so that the best subset is shown on top
    top10 = top10.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.3, 5.4))

    colors = [
        "#dbeaf7", "#d1e5f4", "#c7e0f1", "#bddbee", "#b3d6eb",
        "#a9d1e8", "#9fcc e5".replace(" ", ""), "#95c7e2", "#8bc2df", "#1f4e79"
    ]

    ax.barh(
        top10["subset_label"],
        top10["energy"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Top 10 sensor subsets", pad=10)
    ax.set_xlabel(r"Objective value $J(S)$")
    ax.set_ylabel("Sensor subset")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)

    # annotate only the best subset
    best_row = ranking.iloc[0]
    ax.text(
        best_row["energy"] + 0.015,
        best_row["subset_sensors"].replace("Sensor", "S"),
        f"{best_row['energy']:.3f}",
        va="center",
        ha="left",
        fontsize=11,
        color="#1f4e79"
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()

def plot_top10_breakdown(ranking: pd.DataFrame, save_path: Path | None = None):
    top10 = ranking.head(10).copy()
    x = np.arange(len(top10))
    w = 0.18

    utility_gain = -top10["utility_term"].to_numpy()
    redundancy = top10["redundancy_term"].to_numpy()
    geometry = top10["geometry_term"].to_numpy()
    symmetry = top10["symmetry_term"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    ax.bar(x - 1.5*w, utility_gain, width=w, color="#4f81bd", edgecolor="black", linewidth=0.7, label="Utility gain")
    ax.bar(x - 0.5*w, redundancy,   width=w, color="#9dc3e6", edgecolor="black", linewidth=0.7, label="Redundancy penalty")
    ax.bar(x + 0.5*w, geometry,     width=w, color="#b8cce4", edgecolor="black", linewidth=0.7, label="Geometry penalty")
    ax.bar(x + 1.5*w, symmetry,     width=w, color="#dbe5f1", edgecolor="black", linewidth=0.7, label="Symmetry penalty")

    ax.set_title("Objective-function terms for the top 10 subsets", pad=10)
    ax.set_xlabel("Subset rank")
    ax.set_ylabel("Contribution value")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in top10["rank"]])

    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=True, ncol=2)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()

def plot_top10_sensor_heatmap(ranking: pd.DataFrame, sensor_names: List[str], save_path: Path | None = None):
    top10 = ranking.head(10).copy()

    M = np.zeros((len(top10), len(sensor_names)), dtype=int)

    for r, subset_str in enumerate(top10["subset_sensors"]):
        chosen = [x.strip() for x in subset_str.split(",")]
        for c, s in enumerate(sensor_names):
            if s in chosen:
                M[r, c] = 1

    plt.figure(figsize=(10, 5))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="Selected (1=yes)")
    plt.xticks(np.arange(len(sensor_names)), sensor_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(top10)), [f"Rank {r}" for r in top10["rank"]])
    plt.title("Sensor occurrence in top 10 subsets")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# =============================================================================
# Main brute-force workflow
# =============================================================================
def main():
    qaoa_mod = load_main_module(MAIN_QAOA_FILE)

    # 1) Load data and compute the SAME metrics used by the main QAOA file
    df = qaoa_mod.load_and_prep_data(str(EXCEL_FILE))
    sensor_names = df.columns.tolist()

    b, Corr, diagnostics = qaoa_mod.compute_metrics_from_data(
        df=df,
        sample_rate_hz=SAMPLE_RATE_HZ,
        w_entropy=W_ENTROPY,
        w_spectral=W_SPECTRAL,
        w_fim=W_FIM,
        n_bands=N_SPECTRAL_BANDS,
        w_mi=W_MI,
        w_band_corr=W_BAND_CORR,
        oma_json_path=str(OMA_JSON_FILE),
    )

    # 2) Geometry and symmetry data
    geom_df = qaoa_mod.load_sensor_geometry(str(GEOMETRY_FILE))
    dist_matrix = qaoa_mod.build_distance_matrix(sensor_names, geom_df)
    prox_matrix = qaoa_mod.build_proximity_matrix(dist_matrix, length_scale=GEOM_LENGTH_SCALE)
    left_idx, right_idx, axis_x = qaoa_mod.infer_left_right_groups(
        sensor_names, geom_df, symmetry_coord=SYMMETRY_COORD
    )

    # 3) Build the SAME QUBO matrix used by the optimization
    Q = qaoa_mod.build_qubo_sensor_selection(
        b=b,
        Corr=Corr,
        k=K_SENSORS_TO_SELECT,
        prox_matrix=prox_matrix,
        left_indices=left_idx,
        right_indices=right_idx,
        lam=QUBO_LAMBDA,
        alpha=QUBO_ALPHA,
        gamma=GEOM_GAMMA,
        eta_sym=SYM_ETA,
    )

    constant_offset = QUBO_LAMBDA * (K_SENSORS_TO_SELECT ** 2)

    # 4) Evaluate all C(n, k) subsets
    rows = []
    combos = list(itertools.combinations(range(len(sensor_names)), K_SENSORS_TO_SELECT))

    for combo in combos:
        combo = list(combo)
        combo_names = [sensor_names[i] for i in combo]

        energy = subset_energy_from_Q(Q, combo, constant_offset=constant_offset)
        terms = decompose_subset_terms(
            selected_indices=combo,
            b=b,
            Corr=Corr,
            prox_matrix=prox_matrix,
            left_indices=left_idx,
            right_indices=right_idx,
            alpha=QUBO_ALPHA,
            gamma=GEOM_GAMMA,
            eta_sym=SYM_ETA,
            lam=QUBO_LAMBDA,
            k=K_SENSORS_TO_SELECT,
        )

        rows.append({
            "subset_indices": str(combo),
            "subset_sensors": ", ".join(combo_names),
            "energy": energy,
            **terms,
        })

    ranking = pd.DataFrame(rows).sort_values("energy", ascending=True, ignore_index=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    # 5) Optional comparison with QAOA-selected sensors
    qaoa_rank = None
    qaoa_energy = None
    if QAOA_SELECTED_SENSORS is not None:
        qaoa_set = set(QAOA_SELECTED_SENSORS)
        mask = ranking["subset_sensors"].apply(lambda s: set([x.strip() for x in s.split(",")]) == qaoa_set)
        if mask.any():
            qaoa_row = ranking.loc[mask].iloc[0]
            qaoa_rank = int(qaoa_row["rank"])
            qaoa_energy = float(qaoa_row["energy"])

    # 6) Save files
    ranking.to_csv(RANKING_CSV, index=False)
    ranking.head(10).to_csv(TOP10_CSV, index=False)

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("BRUTE-FORCE SENSOR SELECTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of sensors: {len(sensor_names)}\n")
        f.write(f"Subset size k: {K_SENSORS_TO_SELECT}\n")
        f.write(f"Total combinations tested: {len(combos)}\n")
        f.write(f"Estimated symmetry axis ({SYMMETRY_COORD}): {axis_x:.6f}\n\n")

        f.write("Best subset:\n")
        f.write(f"  Rank: 1\n")
        f.write(f"  Sensors: {ranking.loc[0, 'subset_sensors']}\n")
        f.write(f"  Energy: {ranking.loc[0, 'energy']:.10f}\n")
        f.write(f"  Utility term: {ranking.loc[0, 'utility_term']:.10f}\n")
        f.write(f"  Redundancy term: {ranking.loc[0, 'redundancy_term']:.10f}\n")
        f.write(f"  Geometry term: {ranking.loc[0, 'geometry_term']:.10f}\n")
        f.write(f"  Symmetry term: {ranking.loc[0, 'symmetry_term']:.10f}\n\n")

        if qaoa_rank is not None:
            f.write("Comparison with QAOA subset:\n")
            f.write(f"  QAOA sensors: {', '.join(QAOA_SELECTED_SENSORS)}\n")
            f.write(f"  QAOA rank in brute-force list: {qaoa_rank}\n")
            f.write(f"  QAOA energy: {qaoa_energy:.10f}\n")
            f.write(f"  Global-best energy: {ranking.loc[0, 'energy']:.10f}\n")
            f.write(f"  Energy gap: {qaoa_energy - ranking.loc[0, 'energy']:.10f}\n")

    # 7) Plots
    plot_full_ranking(
       ranking,
       qaoa_selected_sensors=QAOA_SELECTED_SENSORS,
       save_path=RANKING_PNG,
       )
    plot_top10_ranking(
       ranking,
       save_path=TOP10_PNG,
       )
    plot_top10_breakdown(
       ranking,
       save_path=BREAKDOWN_TOP10_PNG,
       )
    plot_top10_sensor_heatmap(
       ranking,
       sensor_names=sensor_names,
       save_path=HEATMAP_TOP10_PNG,
       )   


    # 8) Console summary
    print("\n" + "=" * 70)
    print("BRUTE-FORCE VALIDATION COMPLETED")
    print("=" * 70)
    print(f"Total combinations tested: {len(combos)}")
    print(f"Best subset: {ranking.loc[0, 'subset_sensors']}")
    print(f"Best energy: {ranking.loc[0, 'energy']:.10f}")
    print("\nTop 10 subsets:")
    print(ranking[["rank", "subset_sensors", "energy"]].head(10).to_string(index=False))

    if qaoa_rank is not None:
        print("\nComparison with QAOA subset:")
        print(f"QAOA subset: {', '.join(QAOA_SELECTED_SENSORS)}")
        print(f"Rank in brute-force list: {qaoa_rank}")
        print(f"Energy gap from global optimum: {qaoa_energy - ranking.loc[0, 'energy']:.10f}")

    print(f"\nSaved full ranking to: {RANKING_CSV}")
    print(f"Saved top-10 ranking to: {TOP10_CSV}")
    print(f"Saved summary to: {SUMMARY_TXT}")
    print(f"Saved ranking plot to: {RANKING_PNG}")
    print(f"Saved top-10 plot to: {TOP10_PNG}")
    print(f"Saved breakdown plot to: {BREAKDOWN_TOP10_PNG}")
    print(f"Saved heatmap plot to: {HEATMAP_TOP10_PNG}")


if __name__ == "__main__":
    main()

