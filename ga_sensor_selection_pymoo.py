# -*- coding: utf-8 -*-
"""
Genetic Algorithm benchmark for k-sensor selection using pymoo.

This script is designed to be directly comparable with:
    - qaoa-fm_OMA.py
    - brute_force_sensor_selection.py

It imports the same metric-building and QUBO-building functions from qaoa-fm_OMA.py,
then solves the same fixed-cardinality binary selection problem using a Genetic
Algorithm (GA). The exact cardinality constraint sum(x)=k is enforced through a
custom pymoo Repair operator, so every evaluated solution contains exactly k
selected sensors.

Typical use:
    python ga_sensor_selection_pymoo.py

Required package:
    pip install pymoo
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation


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

# Utility weights: same as qaoa-fm_OMA.py and brute_force_sensor_selection.py
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

# GA parameters
POP_SIZE = 6
N_GENERATIONS = 5
N_RUNS = 4
SEED = 42

# Optional: paste the QAOA solution here to compute direct comparison
QAOA_SELECTED_SENSORS: Optional[List[str]] = None
# Example:
# QAOA_SELECTED_SENSORS = ["Sensor1", "Sensor2", "Sensor8"]

# Output files
OUT_DIR = BASE_DIR / "ga_results"
OUT_DIR.mkdir(exist_ok=True)
RUNS_CSV = OUT_DIR / "ga_runs_summary.csv"
BEST_CSV = OUT_DIR / "ga_best_solution.csv"
CONVERGENCE_PNG = OUT_DIR / "ga_best_convergence.png"
SUMMARY_TXT = OUT_DIR / "ga_summary.txt"


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


def energy_from_binary_vector(x: np.ndarray, Q: np.ndarray, constant_offset: float = 0.0) -> float:
    """Evaluate E(x) = x^T Q x + constant_offset."""
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + constant_offset)


def selected_names_from_x(x: np.ndarray, sensor_names: List[str]) -> List[str]:
    return [sensor_names[i] for i in np.where(np.asarray(x).astype(int) == 1)[0]]


class ExactKRepair(Repair):
    """
    Repair binary individuals so that exactly k variables are active.

    If too many sensors are selected, randomly deactivate the surplus.
    If too few sensors are selected, randomly activate additional sensors.
    This keeps the GA search inside the feasible fixed-cardinality space.
    """

    def __init__(self, k: int, seed: Optional[int] = None):
        super().__init__()
        self.k = int(k)
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X).astype(bool)
        n_individuals, n_var = X.shape

        for r in range(n_individuals):
            active = np.where(X[r])[0]
            inactive = np.where(~X[r])[0]
            n_active = len(active)

            if n_active > self.k:
                to_deactivate = self.rng.choice(active, size=n_active - self.k, replace=False)
                X[r, to_deactivate] = False
            elif n_active < self.k:
                to_activate = self.rng.choice(inactive, size=self.k - n_active, replace=False)
                X[r, to_activate] = True

        return X


class SensorSelectionQUBOProblem(ElementwiseProblem):
    """Single-objective fixed-cardinality binary QUBO problem."""

    def __init__(self, Q: np.ndarray, k: int, constant_offset: float = 0.0):
        self.Q = np.asarray(Q, dtype=float)
        self.k = int(k)
        self.constant_offset = float(constant_offset)
        super().__init__(
            n_var=self.Q.shape[0],
            n_obj=1,
            n_ieq_constr=0,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x).astype(float)
        # The repair operator should enforce feasibility. This small penalty is a safety net.
        cardinality_violation = abs(np.sum(x) - self.k)
        penalty = 1.0e6 * cardinality_violation
        out["F"] = energy_from_binary_vector(x, self.Q, self.constant_offset) + penalty


def build_problem_data():
    """Compute b, Corr, geometry terms and Q using exactly the same workflow as QAOA."""
    qaoa_mod = load_main_module(MAIN_QAOA_FILE)

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

    geom_df = qaoa_mod.load_sensor_geometry(str(GEOMETRY_FILE))
    dist_matrix = qaoa_mod.build_distance_matrix(sensor_names, geom_df)
    prox_matrix = qaoa_mod.build_proximity_matrix(dist_matrix, length_scale=GEOM_LENGTH_SCALE)
    left_idx, right_idx, axis_x = qaoa_mod.infer_left_right_groups(
        sensor_names, geom_df, symmetry_coord=SYMMETRY_COORD
    )

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

    return {
        "Q": Q,
        "constant_offset": constant_offset,
        "sensor_names": sensor_names,
        "b": b,
        "Corr": Corr,
        "diagnostics": diagnostics,
        "axis_x": axis_x,
    }


def run_single_ga(problem: SensorSelectionQUBOProblem, seed: int):
    repair = ExactKRepair(k=K_SENSORS_TO_SELECT, seed=seed)

    algorithm = GA(
        pop_size=POP_SIZE,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9),
        mutation=BitflipMutation(prob=1.0 / problem.n_var),
        repair=repair,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", N_GENERATIONS)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=True,
        verbose=False,
    )

    best_x = np.asarray(result.X).astype(int)
    best_energy = float(result.F[0])

    convergence = []
    for hist in result.history:
        F = hist.pop.get("F")
        convergence.append(float(np.min(F)))

    return best_x, best_energy, convergence


def main():
    data = build_problem_data()
    Q = data["Q"]
    constant_offset = data["constant_offset"]
    sensor_names = data["sensor_names"]

    problem = SensorSelectionQUBOProblem(
        Q=Q,
        k=K_SENSORS_TO_SELECT,
        constant_offset=constant_offset,
    )

    rows = []
    best_overall = None
    best_convergence = None

    for run_id in range(N_RUNS):
        seed = SEED + run_id
        x_best, energy_best, convergence = run_single_ga(problem, seed=seed)
        selected_sensors = selected_names_from_x(x_best, sensor_names)

        row = {
            "run": run_id + 1,
            "seed": seed,
            "energy": energy_best,
            "selected_count": int(np.sum(x_best)),
            "subset_indices": str(np.where(x_best == 1)[0].tolist()),
            "subset_sensors": ", ".join(selected_sensors),
        }
        rows.append(row)

        if best_overall is None or energy_best < best_overall["energy"]:
            best_overall = row.copy()
            best_overall["x"] = x_best.copy()
            best_convergence = convergence

    runs_df = pd.DataFrame(rows).sort_values("energy", ascending=True, ignore_index=True)
    runs_df.insert(0, "rank", np.arange(1, len(runs_df) + 1))
    runs_df.to_csv(RUNS_CSV, index=False)

    best_df = pd.DataFrame([best_overall]).drop(columns=["x"])
    best_df.to_csv(BEST_CSV, index=False)

    # Optional QAOA comparison if the user manually provides the QAOA subset.
    qaoa_energy = None
    qaoa_gap = None
    if QAOA_SELECTED_SENSORS is not None:
        x_qaoa = np.zeros(len(sensor_names), dtype=int)
        name_to_idx = {name: i for i, name in enumerate(sensor_names)}
        for name in QAOA_SELECTED_SENSORS:
            x_qaoa[name_to_idx[name]] = 1
        qaoa_energy = energy_from_binary_vector(x_qaoa, Q, constant_offset)
        qaoa_gap = qaoa_energy - float(best_overall["energy"])

    # Convergence plot for the best run.
    if best_convergence is not None and len(best_convergence) > 0:
        plt.figure(figsize=(8.2, 5.0))
        plt.plot(np.arange(1, len(best_convergence) + 1), best_convergence, marker="o", markersize=2.5)
        plt.xlabel("Generation")
        plt.ylabel(r"Best objective value $J(S)$")
        plt.title("GA convergence for the best run")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(CONVERGENCE_PNG, dpi=400, bbox_inches="tight")
        plt.close()

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("GENETIC ALGORITHM SENSOR SELECTION SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Number of sensors: {len(sensor_names)}\n")
        f.write(f"Subset size k: {K_SENSORS_TO_SELECT}\n")
        f.write(f"Population size: {POP_SIZE}\n")
        f.write(f"Generations: {N_GENERATIONS}\n")
        f.write(f"Independent runs: {N_RUNS}\n\n")
        f.write("Best GA solution:\n")
        f.write(f"  Sensors: {best_overall['subset_sensors']}\n")
        f.write(f"  Energy: {best_overall['energy']:.10f}\n")
        f.write(f"  Selected count: {best_overall['selected_count']}\n\n")
        f.write("Across-run statistics:\n")
        f.write(f"  Best energy: {runs_df['energy'].min():.10f}\n")
        f.write(f"  Mean energy: {runs_df['energy'].mean():.10f}\n")
        f.write(f"  Std energy: {runs_df['energy'].std(ddof=1):.10f}\n")
        f.write(f"  Worst energy: {runs_df['energy'].max():.10f}\n")
        f.write(f"  Unique solutions: {runs_df['subset_sensors'].nunique()}\n")
        if qaoa_energy is not None:
            f.write("\nComparison with QAOA subset:\n")
            f.write(f"  QAOA sensors: {', '.join(QAOA_SELECTED_SENSORS)}\n")
            f.write(f"  QAOA energy: {qaoa_energy:.10f}\n")
            f.write(f"  QAOA - best GA energy gap: {qaoa_gap:.10f}\n")

    print("\n" + "=" * 70)
    print("GENETIC ALGORITHM BENCHMARK COMPLETED")
    print("=" * 70)
    print(f"Best GA subset: {best_overall['subset_sensors']}")
    print(f"Best GA energy: {best_overall['energy']:.10f}")
    print(f"Unique solutions across runs: {runs_df['subset_sensors'].nunique()}")
    print("\nTop GA runs:")
    print(runs_df[["rank", "run", "seed", "subset_sensors", "energy"]].head(10).to_string(index=False))
    print(f"\nSaved run summary to: {RUNS_CSV}")
    print(f"Saved best solution to: {BEST_CSV}")
    print(f"Saved summary to: {SUMMARY_TXT}")
    print(f"Saved convergence plot to: {CONVERGENCE_PNG}")


if __name__ == "__main__":
    main()
