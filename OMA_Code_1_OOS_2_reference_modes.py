"""
Reference modal set of the Mendez-Nunez bridge, used to LABEL modes across records
in the out-of-sample validation (it plays no role in the sensor selection, and the
subset metrics are still computed against the global OMA of each record).

Steps:
    1. Global COV-SSI (10 channels) on REFERENCE_RECORD, with exactly the v1 global
       settings (OMA_Code_1_OOS_1_run_oma.py).
    2. One-to-one assignment of the identified modes to the reference frequencies
       REFERENCE_FREQS_EXP_HZ (nearest frequency first, within REF_FREQ_TOL_PCT).
    3. Save the reference set (frequency, damping, 10-channel mode shape) and its AutoMAC.
    4. Optional: 10-channel MAC consistency of the reference modes in every record
       stored in the v1 modal_results.json (no OMA rerun).

Outputs: results/reference_modes/
"""
import hashlib
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helpers.outils as outils
import OMA_Code_1_OOS_1_run_oma as v1  # frozen OMA settings and shared helpers (forces "Agg")

# ============================================================
# SETTINGS
# ============================================================
REFERENCE_RECORD = "Accelerations_21"

# Reference modes of the bridge (experimental frequencies of the reference figure)
REFERENCE_FREQS_EXP_HZ = [2.87, 3.76, 4.53, 5.05, 5.31, 6.14, 6.42, 6.81, 7.24, 8.31, 9.51]
REF_LABELS = [f"Ref_{k}" for k in range(1, len(REFERENCE_FREQS_EXP_HZ) + 1)]

REF_FREQ_TOL_PCT = 2.5      # [%] labelling tolerance between identified and reference frequencies
REQUIRE_COMPLETE = True     # stop if the record does not identify every reference mode

CHECK_CONSISTENCY = True            # step 4 (reads the v1 modal_results.json)
CONSISTENCY_MAC_THRESHOLD = 0.8     # 10-channel MAC needed to label a record's mode as Ref_k
SANITY_FREQ_TOL_HZ = 1e-3           # recomputed vs stored v1 frequencies of REFERENCE_RECORD

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "reference_modes"


# ============================================================
# HELPERS
# ============================================================
def sha256_of_file(path, chunk_size=1 << 20):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def as_mode_matrix(Phi, n_channels):
    Phi = np.asarray(Phi)
    return Phi.reshape(n_channels, -1) if Phi.ndim == 1 else Phi


def training_length():
    """Training record length from the v1 settings (needed by exclusion criterion (c))."""
    settings_file = v1.OUTPUT_DIR / "settings.json"
    if settings_file.exists():
        with open(settings_file, encoding="utf-8") as f:
            return int(json.load(f)["training_file"]["n_samples"])
    training_file, _ = v1.resolve_paths()
    return v1.load_record(training_file).shape[0]


def assign_to_reference(Fn, f_ref, tol_pct):
    """
    One-to-one assignment of identified modes to reference frequencies: the pair with
    the smallest relative frequency error is assigned first, and so on, while the error
    is within tol_pct. Returns {reference index: identified-mode index}.
    """
    err = 100.0 * np.abs(Fn[None, :] - f_ref[:, None]) / f_ref[:, None]  # (n_ref, n_identified)
    err = np.where(err <= tol_pct, err, np.inf)
    assignment = {}
    while np.isfinite(err).any():
        i, j = np.unravel_index(np.argmin(err), err.shape)
        assignment[int(i)] = int(j)
        err[i, :] = np.inf
        err[:, j] = np.inf
    return assignment


def greedy_mac_assignment(mac, threshold):
    """One-to-one assignment by highest MAC first, while MAC >= threshold. mac: (n_record, n_ref)."""
    work = np.where(np.isfinite(mac), mac, -np.inf)
    assignment = {}
    while work.size and work.max() >= threshold:
        i, j = np.unravel_index(np.argmax(work), work.shape)
        assignment[int(j)] = int(i)  # reference index -> record mode index
        work[i, :] = -np.inf
        work[:, j] = -np.inf
    return assignment


def save_mac_figure(mac, labels, title, filepath):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mac, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    for i in range(mac.shape[0]):
        for j in range(mac.shape[1]):
            ax.text(j, i, f"{mac[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if mac[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="MAC")
    ax.set_title(title)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    f_exp = np.asarray(REFERENCE_FREQS_EXP_HZ, dtype=float)

    # ------------------------------------------------------------
    # 1. Load the reference record and run the global OMA (v1 settings)
    # ------------------------------------------------------------
    training_file, test_files = v1.resolve_paths()
    files = {p.stem: p for p in [training_file, *test_files]}
    if REFERENCE_RECORD not in files:
        raise FileNotFoundError(f"{REFERENCE_RECORD} not found. Available: {list(files)}")
    path = files[REFERENCE_RECORD]

    data = v1.load_record(path)
    reason = v1.check_exclusion(data, training_length())
    if reason is not None:
        raise RuntimeError(f"{REFERENCE_RECORD} fails the exclusion check: {reason}")
    print(f"=== {REFERENCE_RECORD}: {data.shape[0]} samples x {data.shape[1]} channels ===")

    oma_alg, Fn, Phi = v1._run_cov_ssi(data, v1.make_global_clustering())
    v1._save_stab_cluster_plot(
        oma_alg, OUTPUT_DIR / f"{REFERENCE_RECORD}_global_stab_cluster.png",
        f"{REFERENCE_RECORD} | {v1.OMA_METHOD} | Global (10 channels) | Stabilisation clustering",
        plot_noise=True, freqlim=v1.SSI_FREQLIM,
    )

    clus_res = oma_alg.result.clustering_results[v1.CLUSTERING_NAME]
    Xi = getattr(clus_res, "Xi", None)
    Xi = np.ravel(np.asarray(Xi, dtype=float)) if Xi is not None else np.array([])
    if Xi.size != Fn.size:
        Xi = np.full(Fn.size, np.nan)

    order = np.argsort(Fn)
    Fn, Phi, Xi = Fn[order], as_mode_matrix(Phi, data.shape[1])[:, order], Xi[order]
    print(f"Identified global modes ({len(Fn)}): {np.round(Fn, 3).tolist()}")

    # ------------------------------------------------------------
    # 2. Assign the identified modes to the reference frequencies
    # ------------------------------------------------------------
    assignment = assign_to_reference(Fn, f_exp, REF_FREQ_TOL_PCT)
    assigned_modes = set(assignment.values())

    rows = []
    for k, label in enumerate(REF_LABELS):
        j = assignment.get(k)
        rows.append({
            "ref_label": label,
            "f_exp_Hz": f_exp[k],
            "global_mode": f"Mode_{j + 1}" if j is not None else None,
            "f_Hz": Fn[j] if j is not None else np.nan,
            "df_vs_exp_pct": 100.0 * (Fn[j] - f_exp[k]) / f_exp[k] if j is not None else np.nan,
            "xi_pct": 100.0 * Xi[j] if j is not None else np.nan,
        })
    df_ref = pd.DataFrame(rows)
    extra = [f"Mode_{j + 1} ({Fn[j]:.3f} Hz)" for j in range(len(Fn)) if j not in assigned_modes]
    missing = df_ref.loc[df_ref["global_mode"].isna(), "ref_label"].tolist()

    print("\nIdentified vs reference frequencies:")
    print(df_ref.round(3).to_string(index=False))
    print(f"Missing reference modes: {missing}")
    print(f"Identified modes not assigned to any reference mode: {extra}")
    df_ref.to_csv(OUTPUT_DIR / "reference_modes.csv", index=False)

    if missing and REQUIRE_COMPLETE:
        raise SystemExit(f"{REFERENCE_RECORD} does not identify every reference mode "
                         f"(missing: {missing}). Choose another record or set REQUIRE_COMPLETE = False.")

    # ------------------------------------------------------------
    # 3. Reference set: frequencies, damping, 10-channel shapes, AutoMAC
    # ------------------------------------------------------------
    ref_idx = [assignment[k] for k in range(len(REF_LABELS)) if k in assignment]
    ref_labels_found = [REF_LABELS[k] for k in range(len(REF_LABELS)) if k in assignment]
    Fn_ref, Phi_ref, Xi_ref = Fn[ref_idx], Phi[:, ref_idx], Xi[ref_idx]

    auto_mac = np.atleast_2d(outils.MaC(Phi_ref, Phi_ref))
    off = auto_mac[~np.eye(len(ref_idx), dtype=bool)]
    print(f"\nAutoMAC of the reference set: max off-diagonal = {off.max():.3f}, "
          f"mean off-diagonal = {off.mean():.3f}")
    pd.DataFrame(auto_mac, index=ref_labels_found, columns=ref_labels_found).to_csv(
        OUTPUT_DIR / "reference_autoMAC.csv")
    save_mac_figure(auto_mac, ref_labels_found, f"AutoMAC of the reference modes ({REFERENCE_RECORD})",
                    OUTPUT_DIR / "reference_autoMAC.png")

    outils.save_json_serialized({
        "record": REFERENCE_RECORD,
        "ref_labels": ref_labels_found,
        "f_exp_Hz": [float(f_exp[REF_LABELS.index(lab)]) for lab in ref_labels_found],
        "Fn_Hz": Fn_ref,
        "Xi": Xi_ref,
        "Phi": Phi_ref,
        "Phi_real": outils.complex_to_normal_mode(Phi_ref),
        "channels": v1.CHANNELS,
    }, OUTPUT_DIR / "reference_modes.json")

    settings = {
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "reference_record": {"file": path.name, "path": str(path), "sha256": sha256_of_file(path),
                             "n_samples": int(data.shape[0])},
        "oma_settings": "global settings of OMA_Code_1_OOS_1_run_oma.py (see its settings.json)",
        "REFERENCE_FREQS_EXP_HZ": REFERENCE_FREQS_EXP_HZ,
        "REF_FREQ_TOL_PCT": REF_FREQ_TOL_PCT,
        "assignment_rule": "one-to-one, smallest relative frequency error first, within REF_FREQ_TOL_PCT",
        "CONSISTENCY_MAC_THRESHOLD": CONSISTENCY_MAC_THRESHOLD,
        "consistency_rule": "10-channel MAC, one-to-one, highest MAC first, MAC >= threshold",
        "missing_reference_modes": missing,
        "unassigned_identified_modes": extra,
    }
    with open(OUTPUT_DIR / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------
    # 4. Optional: consistency of the reference modes in every record (v1 results)
    # ------------------------------------------------------------
    if not CHECK_CONSISTENCY:
        print(f"\nOutputs written to: {OUTPUT_DIR}")
        return

    modal_v1 = outils.load_json_serialized(v1.OUTPUT_DIR / "modal_results.json")

    # Sanity check: the recomputed OMA must reproduce the stored v1 global modes of this record
    if REFERENCE_RECORD in modal_v1 and "global" in modal_v1[REFERENCE_RECORD]:
        Fn_stored = np.sort(np.ravel(np.asarray(modal_v1[REFERENCE_RECORD]["global"]["Fn_Hz"], dtype=float)))
        same = Fn_stored.size == Fn.size and np.allclose(Fn_stored, Fn, atol=SANITY_FREQ_TOL_HZ)
        print(f"\nRecomputed vs stored v1 global modes of {REFERENCE_RECORD}: "
              f"{'identical' if same else 'DIFFERENT - check the settings'}")

    rows = []
    for record, res in modal_v1.items():
        if "global" not in res:
            continue
        Fn_r = np.ravel(np.asarray(res["global"]["Fn_Hz"], dtype=float))
        Phi_r = as_mode_matrix(res["global"]["Phi"], len(v1.CHANNELS))
        mac = np.atleast_2d(outils.MaC(Phi_r, Phi_ref))  # (n_record_modes, n_ref)
        assigned = greedy_mac_assignment(mac, CONSISTENCY_MAC_THRESHOLD)
        role = "training" if record == training_file.stem else "test"

        for k, label in enumerate(ref_labels_found):
            i = assigned.get(k)
            i_best = int(np.nanargmax(mac[:, k])) if mac.shape[0] else None
            rows.append({
                "role": role,
                "record": record,
                "ref_label": label,
                "f_ref_Hz": Fn_ref[k],
                "assigned_global_mode": f"Mode_{i + 1}" if i is not None else None,
                "f_assigned_Hz": Fn_r[i] if i is not None else np.nan,
                "df_vs_ref_pct": 100.0 * (Fn_r[i] - Fn_ref[k]) / Fn_ref[k] if i is not None else np.nan,
                "MAC_assigned": mac[i, k] if i is not None else np.nan,
                "best_MAC_any": mac[i_best, k] if i_best is not None else np.nan,
                "f_best_any_Hz": Fn_r[i_best] if i_best is not None else np.nan,
            })

    df_cons = pd.DataFrame(rows)
    df_cons.to_csv(OUTPUT_DIR / "consistency_by_record.csv", index=False)

    short = lambda r: r.replace("Accelerations_", "A") if r.startswith("Accelerations_") else "TRAIN"
    df_cons["rec"] = df_cons["record"].map(short)
    pivot_mac = df_cons.pivot(index="rec", columns="ref_label", values="MAC_assigned")[ref_labels_found]
    pivot_best = df_cons.pivot(index="rec", columns="ref_label", values="best_MAC_any")[ref_labels_found]
    n_found = df_cons.groupby("ref_label")["MAC_assigned"].apply(lambda s: int(s.notna().sum()))

    print(f"\n10-channel MAC of each record's assigned mode (blank = no mode with MAC >= "
          f"{CONSISTENCY_MAC_THRESHOLD}):")
    print(pivot_mac.round(3).to_string(na_rep=""))
    print("\nBest 10-channel MAC per reference mode, assigned or not (diagnostic):")
    print(pivot_best.round(3).to_string())
    print(f"\nRecords in which each reference mode is found (out of {df_cons['record'].nunique()}):")
    print(n_found[ref_labels_found].to_string())

    pivot_mac.to_csv(OUTPUT_DIR / "consistency_MAC_assigned.csv")
    pivot_best.to_csv(OUTPUT_DIR / "consistency_MAC_best.csv")
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()