"""
Out-of-sample validation of the FROZEN sensor subsets against the REFERENCE modes
- Mendez-Nunez bridge (v3).

Re-matching only (no OMA is rerun). Inputs:
    results/oos_bridge/modal_results.json                 (OMA_Code_1_OOS_1_run_oma.py)
    results/reference_modes/reference_modes.json          (OMA_Code_1_OOS_2_reference_modes.py)
    results/reference_modes/consistency_MAC_assigned.csv  (OMA_Code_1_OOS_2_reference_modes.py)

For every record (training and test):
    1. Its global modes are labelled Ref_k with the 10-channel MAC to the reference
       modes (one-to-one, highest MAC first, MAC >= REF_LABEL_MAC_THRESHOLD). Global
       modes left unlabelled are non-reference modes: they are logged and stored, and
       excluded from everything that follows.
    2. Each frozen subset is matched to the Ref-labelled global modes only, with the
       frequency-gated greedy matcher of v2 (copied unchanged): global modes first,
       candidates within +-FREQ_TOL_PCT, cost of match_modes_fem_oma_II with the v1
       factors, MAC on the subset rows of the global shapes, same subset prefilter.
    3. Reliable pair: MAC >= MAC_THRESHOLD (|df| <= FREQ_TOL_PCT holds by construction).
    4. Metrics in two versions, computed from the same pairs: "all_ref" (the Ref modes
       present in the record's global OMA) and "robust" (the ROBUST_REF_LABELS present).

Outputs: results/oos_bridge/v3/
"""
import json
import logging
import pathlib
import platform
from datetime import datetime

import numpy as np
import pandas as pd

import helpers.outils as outils
import OMA_Code_1_OOS_1_run_oma as v1                # frozen OMA settings and shared helpers
import OMA_Code_1_OOS_2_reference_modes as refmodes  # reference set and its MAC labelling rule

# ============================================================
# SETTINGS (v3)
# ============================================================
REF_LABEL_MAC_THRESHOLD = 0.8  # 10-channel MAC needed to label a global mode as Ref_k
FREQ_TOL_PCT = 2.0             # [%] matching gate, |df| relative to the global frequency
MAC_THRESHOLD = 0.8            # reliable pair: MAC (subset rows) >= MAC_THRESHOLD

# Reference modes found in all 8 test records (checked against consistency_MAC_assigned.csv)
ROBUST_REF_LABELS = ["Ref_1", "Ref_2", "Ref_3", "Ref_4", "Ref_6", "Ref_7", "Ref_9", "Ref_10", "Ref_11"]
VERSIONS = ["all_ref", "robust"]

MODAL_RESULTS_FILE = v1.OUTPUT_DIR / "modal_results.json"
V1_SETTINGS_FILE = v1.OUTPUT_DIR / "settings.json"
REFERENCE_MODES_FILE = refmodes.OUTPUT_DIR / "reference_modes.json"
REFERENCE_SETTINGS_FILE = refmodes.OUTPUT_DIR / "settings.json"
CONSISTENCY_FILE = refmodes.OUTPUT_DIR / "consistency_MAC_assigned.csv"
OUTPUT_DIR = v1.OUTPUT_DIR / "v3"

# Table 6 (submitted manuscript, Case Study II), copied from TABLE6_SUBSET_FREQS_HZ of v2
# (z_OMA_Code_1_COVSSI_subsets_oos_v2.py): subset frequency [Hz] paired with each training
# global mode (Mode_1..Mode_12); None = reference mode not identified by the subset
TABLE6_SUBSET_FREQS_HZ = {
    "subset_0_1_7": [2.870, 3.813, 4.535, 5.085, 5.310, None, 6.029, 6.479, 6.869, 7.261, 8.283, 9.547],
    "subset_1_2_7": [2.871, 3.814, 4.519, 5.090, 5.281, None, 6.037, 6.509, 6.832, 7.268, 8.287, 9.550],
    "subset_1_2_5": [2.871, 3.816, 4.478, None, 5.182, 5.451, 6.090, 6.439, 6.886, 7.279, 8.295, 9.552],
}
TABLE6_FREQ_TOL_HZ = 1e-3  # Table 6 values are rounded to 3 decimals
TABLE6_NOTES = {"subset_1_2_5": "Table 6 used min_pctg = 0.3 for this subset; the v1 OMA uses 0.2"}

LABELLING_CHECK_ATOL = 1e-9  # v3 labelling vs consistency_MAC_assigned.csv

METRICS = ["n_ref_present", "n_matched", "n_reliable", "frac_reliable",
           "mean_abs_df_pct", "max_abs_df_pct", "MAC_min", "MAC_mean"]
WORST_CASE_METRICS = ["MAC_min", "frac_reliable"]

LABEL_ROW_COLUMNS = [
    "role", "record", "global_mode", "f_global_Hz", "is_reference", "ref_label", "f_ref_Hz",
    "MAC_ref", "best_ref_any", "best_MAC_any",
]
MODE_ROW_COLUMNS = [
    "role", "record", "subset_id", "subset_name", "subset_sensors", "ref_label", "robust",
    "global_mode", "f_global_Hz", "pair_status", "subset_mode", "f_subset_Hz",
    "df_pct", "abs_df_pct", "MAC", "reliable", "rejected_freq_tol",
]
METRIC_ROW_COLUMNS = [
    "role", "record", "subset_id", "subset_name", "subset_sensors", "version", "status", "error",
    "n_global_modes", "n_nonref_global_modes", "n_subset_modes", "n_subset_modes_prefiltered",
    "n_unmatched_subset_modes", *METRICS,
]

logger = logging.getLogger("oos_bridge_v3")


# ============================================================
# SET-UP AND SETTINGS
# ============================================================
def setup_output_and_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(OUTPUT_DIR / "run_log.txt", mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def short_record_name(record):
    """Row label used in consistency_MAC_assigned.csv (OMA_Code_1_OOS_2_reference_modes.py)."""
    return record.replace("Accelerations_", "A") if record.startswith("Accelerations_") else "TRAIN"


def write_settings(v1_settings, ref):
    inputs = {
        "modal_results": MODAL_RESULTS_FILE,
        "v1_settings": V1_SETTINGS_FILE,
        "reference_modes": REFERENCE_MODES_FILE,
        "reference_settings": REFERENCE_SETTINGS_FILE,
        "consistency_MAC_assigned": CONSISTENCY_FILE,
    }
    settings = {
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "REF_LABEL_MAC_THRESHOLD": REF_LABEL_MAC_THRESHOLD,
        "FREQ_TOL_PCT": FREQ_TOL_PCT,
        "MAC_THRESHOLD": MAC_THRESHOLD,
        "ROBUST_REF_LABELS": ROBUST_REF_LABELS,
        "VERSIONS": VERSIONS,
        "labelling": {
            "rule": "10-channel MAC between the record's global modes and the reference modes; "
                    "one-to-one, highest MAC first, MAC >= REF_LABEL_MAC_THRESHOLD",
            "function": "OMA_Code_1_OOS_2_reference_modes.greedy_mac_assignment",
            "non_reference_modes": "global modes left unlabelled; excluded from matching and metrics",
        },
        "matching": {
            "matcher": "gated_match copied unchanged from z_OMA_Code_1_COVSSI_subsets_oos_v2.py",
            "first_input": "Ref-labelled global modes of the record (subset rows), in the record's order",
            "reference": "subset modes after the prefilter",
            "candidates": f"available subset modes with |f_sub - f_glob| / f_glob <= {FREQ_TOL_PCT} %",
            "cost": "match_modes_fem_oma_II (unchanged): "
                    "factor_freq * |f_sub - f_glob| / f_glob + factor_mac * (1 - MAC)",
            "remove_matched_modes": v1.MATCH_REMOVE_MATCHED_MODES,
            "factor_freq": v1.MATCH_FACTOR_FREQ,
            "factor_mac": v1.MATCH_FACTOR_MAC,
            "subset_freq_prefilter": f"keep subset modes with ~(Fn >= {v1.SUBSET_FREQ_PREFILTER_HZ})",
        },
        "reliable_pair": f"matched pair with MAC >= {MAC_THRESHOLD} "
                         f"(|df| <= {FREQ_TOL_PCT} % by construction)",
        "metrics": {
            "all_ref": "over the Ref modes present in the record's global OMA",
            "robust": "over the ROBUST_REF_LABELS present in the record's global OMA",
            "pairs": "both versions use the same pairs (the robust version is not re-matched)",
            "frac_reliable": "n_reliable / n_ref_present",
        },
        "table6": {
            "source": "Quantum_OMA_Submitted_SUBMITTED.pdf, Table 6 (Case Study II), via v2",
            "TABLE6_SUBSET_FREQS_HZ": TABLE6_SUBSET_FREQS_HZ,
            "TABLE6_FREQ_TOL_HZ": TABLE6_FREQ_TOL_HZ,
            "notes": TABLE6_NOTES,
        },
        "reference_modes": {"record": ref["record"], "ref_labels": ref["labels"],
                            "Fn_Hz": [float(f) for f in ref["Fn"]]},
        "inputs": {name: {"path": str(path), "sha256": v1.sha256_of_file(path)}
                   for name, path in inputs.items()},
        "oma_settings_v1": v1_settings,
    }
    with open(OUTPUT_DIR / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ============================================================
# REFERENCE MODES AND LABELLING
# ============================================================
def load_reference_modes():
    data = outils.load_json_serialized(REFERENCE_MODES_FILE)
    labels = [str(label) for label in data["ref_labels"]]
    return {
        "record": data["record"],
        "labels": labels,
        "Fn": np.ravel(np.asarray(data["Fn_Hz"], dtype=float)),
        "Phi": _as_mode_matrix(data["Phi"], len(v1.CHANNELS)),
    }


def robust_labels_from_consistency(test_records):
    """Ref columns of consistency_MAC_assigned.csv with no empty value in any test-record row."""
    df = pd.read_csv(CONSISTENCY_FILE, index_col=0)
    rows = [short_record_name(r) for r in test_records]
    missing = [r for r in rows if r not in df.index]
    if missing:
        raise SystemExit(f"Test records missing from {CONSISTENCY_FILE.name}: {missing}")
    ref_columns = [c for c in df.columns if c.startswith("Ref_")]
    return [c for c in ref_columns if df.loc[rows, c].notna().all()]


def label_global_modes(Fn, Phi, ref):
    """
    Label the global modes with the reference modes (10-channel MAC, one-to-one, highest
    MAC first, MAC >= REF_LABEL_MAC_THRESHOLD). Returns (one row per global mode,
    list with the Ref label of each global mode or None).
    """
    mac = np.atleast_2d(outils.MaC(Phi, ref["Phi"]))  # (n_global, n_ref)
    assignment = refmodes.greedy_mac_assignment(mac, REF_LABEL_MAC_THRESHOLD)  # ref index -> global index
    ref_of_mode = {i: k for k, i in assignment.items()}

    rows, labels = [], []
    for i, f in enumerate(Fn):
        k = ref_of_mode.get(i)
        k_best = int(np.nanargmax(mac[i])) if mac.shape[1] else None
        rows.append({
            "global_mode": f"Mode_{i + 1}",
            "f_global_Hz": float(f),
            "is_reference": k is not None,
            "ref_label": ref["labels"][k] if k is not None else None,
            "f_ref_Hz": float(ref["Fn"][k]) if k is not None else np.nan,
            "MAC_ref": float(mac[i, k]) if k is not None else np.nan,
            "best_ref_any": ref["labels"][k_best] if k_best is not None else None,
            "best_MAC_any": float(mac[i, k_best]) if k_best is not None else np.nan,
        })
        labels.append(ref["labels"][k] if k is not None else None)
    return rows, labels


def compare_with_consistency(df_labels, ref):
    """Differences between the v3 labelling MACs and consistency_MAC_assigned.csv (diagnostic)."""
    df_cons = pd.read_csv(CONSISTENCY_FILE, index_col=0)
    differences = []
    for record, grp in df_labels.groupby("record"):
        row = short_record_name(record)
        mac_v3 = grp.dropna(subset=["ref_label"]).set_index("ref_label")["MAC_ref"]
        for label in ref["labels"]:
            a = float(mac_v3.get(label, np.nan))
            b = float(df_cons.loc[row, label]) if row in df_cons.index and label in df_cons.columns else np.nan
            if not np.isclose(a, b, rtol=0.0, atol=LABELLING_CHECK_ATOL, equal_nan=True):
                differences.append(f"{record} {label}: v3 {a:.6f} vs csv {b:.6f}")
    return differences


# ============================================================
# MATCHING: frequency-gated greedy matcher copied unchanged from v2
# (z_OMA_Code_1_COVSSI_subsets_oos_v2.py). Inside these three functions "reference"
# means the second input of the matcher, i.e. the subset modes.
# ============================================================
def _as_mode_matrix(Phi, n_channels):
    Phi = np.asarray(Phi)
    return Phi.reshape(n_channels, -1) if Phi.ndim == 1 else Phi


def _best_reference_mode(f, phi, Fn_ref, Phi_ref, pool):
    """Best reference mode within `pool`, using match_modes_fem_oma_II (unchanged) for the cost."""
    matched_info, ref_labels = outils.match_modes_fem_oma_II(
        frequencies_oma=np.array([f]),
        modes_oma=phi[:, np.newaxis],
        frequencies_model=Fn_ref[pool],
        modes_model=Phi_ref[:, pool],
        oma_labels=["mode"],
        remove_matched_modes=v1.MATCH_REMOVE_MATCHED_MODES,
        factor_freq=v1.MATCH_FACTOR_FREQ,
        factor_mac=v1.MATCH_FACTOR_MAC,
        label_FEM="REF",
    )
    k = next(idx for idx, label in enumerate(ref_labels) if label is not None)
    return pool[k], float(matched_info["mode"]["MAC"])


def gated_match(Fn_first, Phi_first, Fn_ref, Phi_ref, tol_pct):
    """
    Frequency-gated greedy matching. The first-input modes are visited in order; for
    each one, the candidates are the reference modes still available within tol_pct
    of its frequency, and the best candidate (match_modes_fem_oma_II cost) is assigned
    and removed. A mode with no candidate stays unmatched; for traceability, the
    reference mode the ungated rule would have chosen is returned with matched = False.
    Returns (one assignment per first-input mode, indices of unassigned reference modes).
    """
    available = list(range(len(Fn_ref)))
    assignments = []
    for i, f in enumerate(Fn_first):
        candidates = [j for j in available if 100.0 * abs(Fn_ref[j] - f) / f <= tol_pct]
        pool = candidates or available
        if not pool:
            assignments.append({"ref_index": None, "MAC": np.nan, "matched": False})
            continue
        j, mac = _best_reference_mode(f, Phi_first[:, i], Fn_ref, Phi_ref, pool)
        if candidates:
            available.remove(j)
        assignments.append({"ref_index": j, "MAC": mac, "matched": bool(candidates)})
    return assignments, available


def match_subset_to_reference(Fn_subset, Phi_subset, Fn_ref_global, Phi_ref_global, subset):
    """
    v2 matching restricted to the Ref-labelled global modes: these go first (restricted
    to the subset rows) and the prefiltered subset modes are the candidates.
    Returns (assignments, prefiltered subset frequencies, unassigned subset-mode indices).
    """
    matching_mask = ~(Fn_subset >= v1.SUBSET_FREQ_PREFILTER_HZ)
    Fn_subset_match = Fn_subset[matching_mask]
    Phi_subset_match = Phi_subset[:, matching_mask]
    assignments, unassigned = gated_match(
        Fn_ref_global, Phi_ref_global[subset, :], Fn_subset_match, Phi_subset_match, FREQ_TOL_PCT)
    return assignments, Fn_subset_match, unassigned


def compute_metrics(mode_rows):
    """Metrics over the Ref modes in mode_rows (one row per Ref mode present)."""
    n_ref_present = len(mode_rows)
    matched = [r for r in mode_rows if r["pair_status"] == "matched"]
    n_matched = len(matched)
    n_reliable = int(sum(r["reliable"] for r in matched))
    abs_df = np.array([r["abs_df_pct"] for r in matched], dtype=float)
    macs = np.array([r["MAC"] for r in matched], dtype=float)
    return {
        "n_ref_present": n_ref_present,
        "n_matched": n_matched,
        "n_reliable": n_reliable,
        "frac_reliable": n_reliable / n_ref_present if n_ref_present else np.nan,
        "mean_abs_df_pct": abs_df.mean() if n_matched else np.nan,
        "max_abs_df_pct": abs_df.max() if n_matched else np.nan,
        "MAC_min": macs.min() if n_matched else np.nan,
        "MAC_mean": macs.mean() if n_matched else np.nan,
    }


# ============================================================
# ONE RECORD
# ============================================================
def process_record(record, modal, role, ref):
    out = {"global_ok": False, "label_rows": [], "mode_rows": [], "metric_rows": [], "nonref_rows": []}

    def failed_rows(meta, status, error, n_present):
        return [{"role": role, "record": record, **meta, "version": version, "status": status,
                 "error": error, "n_ref_present": n_present[version]} for version in VERSIONS]

    if "global" not in modal:
        logger.error(f"[{record}] no global OMA result in {MODAL_RESULTS_FILE.name}")
        for subset_id, subset in enumerate(v1.SUBSETS, start=1):
            out["metric_rows"] += failed_rows(v1.subset_meta(subset_id, subset), "global_oma_failed",
                                              "no global OMA result in v1", dict.fromkeys(VERSIONS))
        return out

    Fn_global = np.ravel(np.asarray(modal["global"]["Fn_Hz"], dtype=float))
    Phi_global = _as_mode_matrix(modal["global"]["Phi"], len(v1.CHANNELS))
    out["global_ok"] = True

    # 1. Reference labelling of the global modes
    label_rows, labels = label_global_modes(Fn_global, Phi_global, ref)
    out["label_rows"] = [{"role": role, "record": record, **row} for row in label_rows]
    out["nonref_rows"] = [row for row in out["label_rows"] if not row["is_reference"]]
    ref_idx = [i for i, label in enumerate(labels) if label is not None]
    n_present = {"all_ref": len(ref_idx),
                 "robust": sum(labels[i] in ROBUST_REF_LABELS for i in ref_idx)}

    nonref_text = ", ".join(f"{r['global_mode']} ({r['f_global_Hz']:.3f} Hz, best {r['best_ref_any']} "
                            f"MAC {r['best_MAC_any']:.3f})" for r in out["nonref_rows"]) or "none"
    logger.info(f"[{record}] {len(Fn_global)} global modes: {n_present['all_ref']} Ref-labelled "
                f"({n_present['robust']} robust); non-reference: {nonref_text}")

    # 2-4. Matching of each subset to the Ref-labelled global modes, and metrics
    for subset_id, subset in enumerate(v1.SUBSETS, start=1):
        meta = v1.subset_meta(subset_id, subset)
        base = {"role": role, "record": record, **meta, "n_global_modes": len(Fn_global),
                "n_nonref_global_modes": len(out["nonref_rows"])}

        if meta["subset_name"] not in modal:
            logger.error(f"[{record}] {meta['subset_name']}: no subset OMA result in v1")
            out["metric_rows"] += [{**base, **row} for row in
                                   failed_rows(meta, "oma_failed", "no subset OMA result in v1", n_present)]
            continue

        Fn_subset = np.ravel(np.asarray(modal[meta["subset_name"]]["Fn_Hz"], dtype=float))
        Phi_subset = _as_mode_matrix(modal[meta["subset_name"]]["Phi"], len(subset))
        try:
            assignments, Fn_subset_match, unassigned = match_subset_to_reference(
                Fn_subset, Phi_subset, Fn_global[ref_idx], Phi_global[:, ref_idx], subset)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"[{record}] {meta['subset_name']}: matching_failed: {error}")
            out["metric_rows"] += [{**base, **row} for row in
                                   failed_rows(meta, "matching_failed", error, n_present)]
            continue

        mode_rows = []
        for assignment, i in zip(assignments, ref_idx):
            f_global = float(Fn_global[i])
            row = {"role": role, "record": record, **meta, "ref_label": labels[i],
                   "robust": labels[i] in ROBUST_REF_LABELS, "global_mode": f"Mode_{i + 1}",
                   "f_global_Hz": f_global, "reliable": False, "rejected_freq_tol": False}
            j = assignment["ref_index"]
            if j is None:
                row["pair_status"] = "no_subset_mode_left"
            else:
                f_subset = float(Fn_subset_match[j])
                df_pct = 100.0 * (f_subset - f_global) / f_global
                row.update(subset_mode=f"Subset_Mode_{j + 1}", f_subset_Hz=f_subset,
                           df_pct=df_pct, abs_df_pct=abs(df_pct), MAC=assignment["MAC"])
                if assignment["matched"]:
                    if abs(df_pct) > FREQ_TOL_PCT + 1e-9:  # cannot happen with the gated matcher
                        raise RuntimeError(f"[{record}] {meta['subset_name']}: matched pair outside the gate")
                    row.update(pair_status="matched", reliable=bool(assignment["MAC"] >= MAC_THRESHOLD))
                else:
                    # Traceability only: nearest available subset mode, outside the tolerance
                    row.update(pair_status="rejected_freq_tol", rejected_freq_tol=True)
            mode_rows.append(row)
        out["mode_rows"] += mode_rows

        counts = {"n_subset_modes": len(Fn_subset), "n_subset_modes_prefiltered": len(Fn_subset_match),
                  "n_unmatched_subset_modes": len(unassigned)}
        for version in VERSIONS:
            rows_v = mode_rows if version == "all_ref" else [r for r in mode_rows if r["robust"]]
            out["metric_rows"].append({**base, "version": version, "status": "ok", "error": "",
                                       **counts, **compute_metrics(rows_v)})

        m = out["metric_rows"][-2]  # all_ref
        logger.info(f"[{record}] {meta['subset_name']} ({meta['subset_sensors']}): {m['n_ref_present']} Ref "
                    f"modes, {len(Fn_subset_match)} subset modes -> {m['n_matched']} matched, "
                    f"{m['n_reliable']} reliable, {len(unassigned)} subset modes unpaired; "
                    f"MAC_min = {m['MAC_min']:.3f}, max|df| = {m['max_abs_df_pct']:.2f} %")
    return out


# ============================================================
# AGGREGATION AND TABLES
# ============================================================
def aggregate(df_test):
    """Per subset and version: median, Q1, Q3, min and max of every metric, plus worst cases."""
    rows = []
    for subset_id, subset in enumerate(v1.SUBSETS, start=1):
        meta = v1.subset_meta(subset_id, subset)
        for version in VERSIONS:
            grp = df_test[(df_test["subset_name"] == meta["subset_name"]) & (df_test["version"] == version)]
            row = {**meta, "version": version, "n_records": grp["record"].nunique(),
                   "n_ok": int((grp["status"] == "ok").sum())}
            for metric in METRICS:
                values = grp[metric].astype(float).dropna().to_numpy()
                if values.size:
                    q1, median, q3 = np.percentile(values, [25, 50, 75])
                    stats = {"median": median, "Q1": q1, "Q3": q3, "min": values.min(), "max": values.max()}
                else:
                    stats = dict.fromkeys(["median", "Q1", "Q3", "min", "max"], np.nan)
                row.update({f"{metric}_{stat}": value for stat, value in stats.items()})
            for metric in WORST_CASE_METRICS:
                valid = grp.dropna(subset=[metric])
                if valid.empty:
                    row.update({f"{metric}_worst_case": np.nan, f"{metric}_worst_case_records": ""})
                    continue
                values = valid[metric].astype(float)
                worst = values.min()
                records = valid.loc[np.isclose(values, worst, rtol=0.0, atol=1e-12), "record"]
                row.update({f"{metric}_worst_case": worst, f"{metric}_worst_case_records": ";".join(records)})
            rows.append(row)
    return pd.DataFrame(rows)


def ref_mode_recovery(df_modes_test, df_labels_test, ref, n_test_records):
    """Per subset and Ref mode, over the test records."""
    rows = []
    for subset_id, subset in enumerate(v1.SUBSETS, start=1):
        meta = v1.subset_meta(subset_id, subset)
        sub = df_modes_test[df_modes_test["subset_name"] == meta["subset_name"]]
        for k, label in enumerate(ref["labels"]):
            matched = sub[(sub["ref_label"] == label) & (sub["pair_status"] == "matched")]
            rows.append({
                **meta,
                "ref_label": label,
                "f_ref_Hz": float(ref["Fn"][k]),
                "robust": label in ROBUST_REF_LABELS,
                "n_test_records": n_test_records,
                "n_records_present_in_global": df_labels_test.loc[
                    df_labels_test["ref_label"] == label, "record"].nunique(),
                "n_records_matched": matched["record"].nunique(),
                "n_records_recovered_reliable": matched.loc[matched["reliable"].astype(bool), "record"].nunique(),
                "median_MAC_matched": matched["MAC"].astype(float).median() if not matched.empty else np.nan,
            })
    return pd.DataFrame(rows)


def training_in_sample_table(df_modes_train, df_metrics_train):
    """Table-6 style: one row per Ref mode present in the training record, then summary rows."""
    metas = [v1.subset_meta(k, s) for k, s in enumerate(v1.SUBSETS, start=1)]
    tags = [m["subset_sensors"] for m in metas]
    columns = ["row_type", "version", "item", "robust", "global_mode", "f_global_Hz"]
    for tag in tags:
        columns += [f"f_subset_Hz_{tag}", f"df_pct_{tag}", f"MAC_{tag}", f"reliable_{tag}", f"value_{tag}"]

    rows = []
    ref_rows = df_modes_train.drop_duplicates("ref_label").sort_values("f_global_Hz")
    for ref_row in ref_rows.itertuples():
        row = {"row_type": "mode", "version": "", "item": ref_row.ref_label, "robust": ref_row.robust,
               "global_mode": ref_row.global_mode, "f_global_Hz": ref_row.f_global_Hz}
        for meta, tag in zip(metas, tags):
            pair = df_modes_train[(df_modes_train["subset_name"] == meta["subset_name"])
                                  & (df_modes_train["ref_label"] == ref_row.ref_label)
                                  & (df_modes_train["pair_status"] == "matched")]
            if not pair.empty:
                p = pair.iloc[0]
                row.update({f"f_subset_Hz_{tag}": p["f_subset_Hz"], f"df_pct_{tag}": p["df_pct"],
                            f"MAC_{tag}": p["MAC"], f"reliable_{tag}": bool(p["reliable"])})
        rows.append(row)

    for version in VERSIONS:
        for metric in METRICS:
            row = {"row_type": "summary", "version": version, "item": metric}
            for meta, tag in zip(metas, tags):
                m = df_metrics_train[(df_metrics_train["subset_name"] == meta["subset_name"])
                                     & (df_metrics_train["version"] == version)]
                row[f"value_{tag}"] = m[metric].iloc[0] if not m.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def training_vs_table6(df_labels_train, df_modes_train):
    """
    Informative only: every training pairing that differs from Table 6, and every Table 6
    row excluded because its global mode is not a reference mode.
    """
    labels = df_labels_train.set_index("global_mode")
    rows = []
    for subset_id, subset in enumerate(v1.SUBSETS, start=1):
        meta = v1.subset_meta(subset_id, subset)
        expected = TABLE6_SUBSET_FREQS_HZ.get(meta["subset_name"])
        if expected is None:
            continue
        pairs = df_modes_train[(df_modes_train["subset_name"] == meta["subset_name"])
                               & (df_modes_train["pair_status"] == "matched")].set_index("global_mode")
        for k, f_table6 in enumerate(expected, start=1):
            global_mode = f"Mode_{k}"
            label = labels.loc[global_mode] if global_mode in labels.index else None
            pair = pairs.loc[global_mode] if global_mode in pairs.index else None
            f_v3 = float(pair["f_subset_Hz"]) if pair is not None else np.nan

            if label is None or not bool(label["is_reference"]):
                status = "excluded_non_reference"
            elif f_table6 is None and pair is None:
                continue
            elif f_table6 is None:
                status = "now_matched"
            elif pair is None:
                status = "now_unmatched"
            elif abs(f_v3 - f_table6) > TABLE6_FREQ_TOL_HZ:
                status = "different_subset_mode"
            else:
                continue

            rows.append({
                **meta,
                "table6_mode": global_mode,
                "f_global_Hz": float(label["f_global_Hz"]) if label is not None else np.nan,
                "ref_label": label["ref_label"] if label is not None else None,
                "best_ref_any": label["best_ref_any"] if label is not None else None,
                "best_MAC_any": float(label["best_MAC_any"]) if label is not None else np.nan,
                "f_subset_table6_Hz": np.nan if f_table6 is None else f_table6,
                "f_subset_v3_Hz": f_v3,
                "MAC_v3": float(pair["MAC"]) if pair is not None else np.nan,
                "reliable_v3": bool(pair["reliable"]) if pair is not None else False,
                "status": status,
                "note": TABLE6_NOTES.get(meta["subset_name"], ""),
            })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    setup_output_and_logging()

    with open(V1_SETTINGS_FILE, encoding="utf-8") as f:
        v1_settings = json.load(f)
    modal_results = outils.load_json_serialized(MODAL_RESULTS_FILE)
    ref = load_reference_modes()
    write_settings(v1_settings, ref)

    training_record = pathlib.Path(v1_settings["training_file"]["file"]).stem
    test_records = sorted((r for r in modal_results if r != training_record),
                          key=lambda r: v1._record_number(pathlib.Path(r)))
    logger.info(f"Reference modes ({ref['record']}): "
                + ", ".join(f"{lab} {f:.3f}" for lab, f in zip(ref["labels"], ref["Fn"])))
    logger.info(f"Matching: frequency-gated greedy (v2), Ref-labelled global modes first, "
                f"FREQ_TOL_PCT = {FREQ_TOL_PCT} %, MAC_THRESHOLD = {MAC_THRESHOLD}")

    # ------------------------------------------------------------
    # 1. ROBUST_REF_LABELS must match the consistency file
    # ------------------------------------------------------------
    robust_from_file = robust_labels_from_consistency(test_records)
    if set(robust_from_file) != set(ROBUST_REF_LABELS):
        logger.error(f"ROBUST_REF_LABELS {ROBUST_REF_LABELS} do not match the Ref columns complete in all "
                     f"test records of {CONSISTENCY_FILE.name}: {robust_from_file}")
        raise SystemExit(1)
    logger.info(f"ROBUST_REF_LABELS match {CONSISTENCY_FILE.name}: {robust_from_file}")

    # ------------------------------------------------------------
    # 2. Records: labelling, matching and metrics
    # ------------------------------------------------------------
    results = {training_record: process_record(training_record, modal_results[training_record], "training", ref)}
    for record in test_records:
        results[record] = process_record(record, modal_results[record], "test", ref)

    df_labels = pd.DataFrame([r for res in results.values() for r in res["label_rows"]], columns=LABEL_ROW_COLUMNS)
    df_modes = pd.DataFrame([r for res in results.values() for r in res["mode_rows"]], columns=MODE_ROW_COLUMNS)
    df_metrics = pd.DataFrame([r for res in results.values() for r in res["metric_rows"]],
                              columns=METRIC_ROW_COLUMNS)
    df_labels.to_csv(OUTPUT_DIR / "global_ref_labelling.csv", index=False)
    df_modes.to_csv(OUTPUT_DIR / "per_mode_matching.csv", index=False)
    df_metrics.to_csv(OUTPUT_DIR / "per_record_subset_metrics.csv", index=False)

    differences = compare_with_consistency(df_labels, ref)
    if differences:
        logger.warning("Labelling differs from consistency_MAC_assigned.csv:\n    " + "\n    ".join(differences))
    else:
        logger.info("Labelling identical to consistency_MAC_assigned.csv (all records, all Ref modes).")

    # ------------------------------------------------------------
    # 3. Aggregation (test records) and training tables
    # ------------------------------------------------------------
    processed_tests = [r for r in test_records if results[r]["global_ok"]]
    df_test = df_metrics[df_metrics["role"] == "test"]
    df_aggregated = aggregate(df_test)
    df_aggregated.to_csv(OUTPUT_DIR / "aggregated_metrics.csv", index=False)

    df_recovery = ref_mode_recovery(df_modes[df_modes["role"] == "test"],
                                    df_labels[df_labels["role"] == "test"], ref, len(processed_tests))
    df_recovery.to_csv(OUTPUT_DIR / "ref_mode_recovery.csv", index=False)

    df_modes_train = df_modes[df_modes["role"] == "training"]
    df_labels_train = df_labels[df_labels["role"] == "training"]
    training_in_sample_table(df_modes_train, df_metrics[df_metrics["role"] == "training"]).to_csv(
        OUTPUT_DIR / "training_in_sample.csv", index=False)
    df_table6 = training_vs_table6(df_labels_train, df_modes_train)
    df_table6.to_csv(OUTPUT_DIR / "training_vs_table6.csv", index=False)

    # ------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------
    failures = df_metrics[(df_metrics["status"] != "ok") & (df_metrics["version"] == "all_ref")]
    lines = ["", "=" * 60, "SUMMARY (v3)", "=" * 60,
             f"Records processed: training ({training_record}) + {len(processed_tests)} test "
             f"({', '.join(processed_tests)})",
             f"Failures (global OMA / subset OMA / matching): {len(failures)}"]
    lines += [f"    - {r.record} | {r.subset_name}: {r.status} {r.error}" for r in failures.itertuples()]
    lines += ["", "Non-reference global modes (excluded):"]
    for record, res in results.items():
        text = ", ".join(f"{r['f_global_Hz']:.3f} Hz (best {r['best_ref_any']}, MAC {r['best_MAC_any']:.3f})"
                         for r in res["nonref_rows"]) or "none"
        present = sorted({r["ref_label"] for r in res["label_rows"] if r["is_reference"]},
                         key=lambda lab: int(lab.split("_")[1]))
        missing = [lab for lab in ref["labels"] if lab not in present]
        lines.append(f"    {record}: {text} | Ref missing: {', '.join(missing) or 'none'}")

    for version in VERSIONS:
        lines += ["", f"Test records, version '{version}': median [Q1, Q3] and worst case"]
        for r in df_aggregated[df_aggregated["version"] == version].itertuples():
            lines.append(
                f"    {r.subset_name} ({r.subset_sensors}): "
                f"n_reliable/n_ref = {r.n_reliable_median:.1f}/{r.n_ref_present_median:.1f}, "
                f"frac_reliable = {r.frac_reliable_median:.3f} [{r.frac_reliable_Q1:.3f}, {r.frac_reliable_Q3:.3f}] "
                f"worst {r.frac_reliable_worst_case:.3f} ({r.frac_reliable_worst_case_records}); "
                f"MAC_min = {r.MAC_min_median:.3f} [{r.MAC_min_Q1:.3f}, {r.MAC_min_Q3:.3f}] "
                f"worst {r.MAC_min_worst_case:.3f} ({r.MAC_min_worst_case_records}); "
                f"max|df| = {r.max_abs_df_pct_max:.2f} %"
            )

    lines += ["", "Training record vs Table 6 (informative):"]
    if df_table6.empty:
        lines.append("    no differences")
    else:
        for (name, sensors), grp in df_table6.groupby(["subset_name", "subset_sensors"], sort=False):
            lines.append(f"    {sensors}: " + ", ".join(f"{s} x{n}" for s, n in grp["status"].value_counts().items()))
    lines.append(f"\nOutputs written to: {OUTPUT_DIR}")
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
