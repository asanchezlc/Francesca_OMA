"""
Out-of-sample validation of the FROZEN sensor subsets - Mendez-Nunez bridge (v2).

Re-matching only: the OMA results of OMA_Code_1_OOS_1_run_oma.py are read
from results/oos_bridge/modal_results.json (no OMA is rerun) and matched again:
    A. Frequency-gated greedy matching, global modes first. The global modes of a
       record are visited in order; each one is assigned its best subset mode among
       the still-available subset modes within +-FREQ_TOL_PCT of its frequency, with
       the cost of helpers.outils.match_modes_fem_oma_II (called unchanged, same
       arguments). A global mode with no candidate stays unmatched; extra subset
       modes stay unmatched.
    B. Reliable mode: matched pair with MAC >= MAC_THRESHOLD and |df| <= FREQ_TOL_PCT.
Before the test records, the training-record pairings of {S1,S2,S8} and {S2,S3,S8}
must reproduce Table 6 of the submitted manuscript; otherwise the script stops.
The ranking is computed only over test records where all three subsets have valid
results. Outputs are written to results/oos_bridge/v2/ (v1 outputs are kept).
"""
import json
import logging
import pathlib
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

import helpers.outils as outils
import OMA_Code_1_OOS_1_run_oma as v1  # frozen settings and shared helpers of the OMA run

# ============================================================
# SETTINGS (v2)
# ============================================================
FREQ_TOL_PCT = 2.0  # [%] matching gate AND reliability limit on |df| (df relative to the global frequency)

V1_DIR = v1.OUTPUT_DIR
MODAL_RESULTS_FILE = V1_DIR / "modal_results.json"
OUTPUT_DIR = V1_DIR / "v2"

# Table 6 (submitted manuscript, Case Study II): subset frequency [Hz] paired with each
# training global mode (Mode_1..Mode_12); None = reference mode not identified by the subset
TABLE6_SUBSET_FREQS_HZ = {
    "subset_0_1_7": [2.870, 3.813, 4.535, 5.085, 5.310, None, 6.029, 6.479, 6.869, 7.261, 8.283, 9.547],
    "subset_1_2_7": [2.871, 3.814, 4.519, 5.090, 5.281, None, 6.037, 6.509, 6.832, 7.268, 8.287, 9.550],
    "subset_1_2_5": [2.871, 3.816, 4.478, None, 5.182, 5.451, 6.090, 6.439, 6.886, 7.279, 8.295, 9.552],
}
TABLE6_GATING_SUBSETS = ["subset_0_1_7", "subset_1_2_7"]  # {S1,S2,S8}, {S2,S3,S8}; {S2,S3,S6} informative only
TABLE6_FREQ_TOL_HZ = 1e-3  # Table 6 values are rounded to 3 decimals

DF_DIAGNOSTIC_PCT = 2.0  # summary only: matched pairs with DF_DIAGNOSTIC_PCT < |df| <= FREQ_TOL_PCT

SUBSET_ROW_COLUMNS = [
    "role", "record", "subset_id", "subset_name", "subset_sensors", "status", "error",
    "n_global_modes", "n_subset_modes", "n_subset_modes_prefiltered",
    "n_matched", "n_rejected_freq_tol", "n_unmatched_subset_modes", "n_reliable", "frac_reliable",
    "mean_abs_df_pct", "max_abs_df_pct", "MAC_min", "MAC_mean",
]
MODE_ROW_COLUMNS = [
    "role", "record", "subset_id", "subset_name", "subset_sensors", "pair_status",
    "global_mode", "f_global_Hz", "training_mode", "f_training_Hz", "tracking_MAC",
    "subset_mode", "f_subset_Hz", "df_pct", "abs_df_pct", "MAC",
    "reliable", "rejected_freq_tol", "closer_global_mode",
]
AGGREGATED_METRICS = [
    "n_global_modes", "n_matched", "n_rejected_freq_tol", "n_unmatched_subset_modes",
    "n_reliable", "frac_reliable", "mean_abs_df_pct", "max_abs_df_pct", "MAC_min", "MAC_mean",
]

logger = logging.getLogger("oos_bridge_v2")


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


def write_settings(v1_settings, modal_results_sha256):
    """v1 settings (OMA, subsets, exclusion, test-file hashes) extended with the v2 matching rules."""
    settings = dict(v1_settings)
    settings["oma_run_started"] = settings.pop("run_started")
    settings["v2_run_started"] = datetime.now().isoformat(timespec="seconds")
    settings["oma_source"] = {
        "file": str(MODAL_RESULTS_FILE),
        "sha256": modal_results_sha256,
        "note": "OMA not rerun in v2; modes read from the v1 run",
    }
    settings["FREQ_TOL_PCT"] = FREQ_TOL_PCT
    settings["matching"] = {
        "rule": "frequency-gated greedy matching, global modes first",
        "first_input": "global modes of the record, restricted to the subset rows, visited in order",
        "reference": "subset modes after the prefilter",
        "candidates": f"available subset modes with |f_sub - f_glob| / f_glob <= {FREQ_TOL_PCT} %",
        "selection": "helpers.outils.match_modes_fem_oma_II (unchanged) on the candidate set",
        "cost": "factor_freq * |f_sub - f_glob| / f_glob + factor_mac * (1 - MAC)",
        "remove_matched_modes": v1.MATCH_REMOVE_MATCHED_MODES,
        "factor_freq": v1.MATCH_FACTOR_FREQ,
        "factor_mac": v1.MATCH_FACTOR_MAC,
        "subset_freq_prefilter": f"keep subset modes with ~(Fn >= {v1.SUBSET_FREQ_PREFILTER_HZ})",
        "no_candidate": "global mode left unmatched (pair_status rejected_freq_tol); the subset mode "
                        "the ungated rule would have chosen is reported for traceability only",
    }
    settings["reliable_mode"] = (f"matched pair with MAC >= {v1.MAC_THRESHOLD} and |df| <= {FREQ_TOL_PCT} % "
                                 "(df relative to the global frequency)")
    settings["ranking"] = {
        "criteria": v1.RANKING_CRITERIA,
        "TIE_ATOL": v1.TIE_ATOL,
        "rule": "only test records where all three subsets have status ok and finite MAC_min and "
                "frac_reliable; unique best = win, shared best = tie",
    }
    settings["table6_check"] = {
        "source": "Quantum_OMA_Submitted_SUBMITTED.pdf, Table 6 (Case Study II)",
        "TABLE6_SUBSET_FREQS_HZ": TABLE6_SUBSET_FREQS_HZ,
        "gating_subsets": TABLE6_GATING_SUBSETS,
        "TABLE6_FREQ_TOL_HZ": TABLE6_FREQ_TOL_HZ,
    }
    with open(OUTPUT_DIR / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ============================================================
# MATCHING (frequency-gated; helpers unchanged)
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


def _closer_global_mode(f_subset, Fn_global, i_assigned):
    """Label of a different global mode closer in frequency to f_subset, or '' (diagnostic)."""
    k = int(np.argmin(np.abs(Fn_global - f_subset) / Fn_global))
    return f"Mode_{k + 1}" if k != i_assigned else ""


def match_subset_to_global(Fn_subset, Phi_subset, Fn_global, Phi_global, subset):
    """
    Gated matching with the global modes (restricted to the subset rows) as first input
    and the prefiltered subset modes as reference. Returns (mode_rows, n_prefiltered, error).
    """
    matching_mask = ~(Fn_subset >= v1.SUBSET_FREQ_PREFILTER_HZ)
    Fn_subset_match = Fn_subset[matching_mask]
    Phi_subset_match = Phi_subset[:, matching_mask]
    Phi_global_subset = Phi_global[subset, :]

    try:
        assignments, unassigned = gated_match(Fn_global, Phi_global_subset, Fn_subset_match, Phi_subset_match,
                                              FREQ_TOL_PCT)
    except Exception as exc:
        return [], len(Fn_subset_match), f"{type(exc).__name__}: {exc}"

    mode_rows = []
    for i, assignment in enumerate(assignments):
        f_global = float(Fn_global[i])
        row = {"global_mode": f"Mode_{i + 1}", "f_global_Hz": f_global,
               "reliable": False, "rejected_freq_tol": False}
        j = assignment["ref_index"]
        if j is None:
            row["pair_status"] = "no_subset_mode_left"
        else:
            f_subset = float(Fn_subset_match[j])
            df_pct = 100.0 * (f_subset - f_global) / f_global
            row.update(subset_mode=f"Subset_Mode_{j + 1}", f_subset_Hz=f_subset,
                       df_pct=df_pct, abs_df_pct=abs(df_pct), MAC=assignment["MAC"])
            if assignment["matched"]:
                row.update(pair_status="matched",
                           reliable=bool(assignment["MAC"] >= v1.MAC_THRESHOLD and abs(df_pct) <= FREQ_TOL_PCT),
                           closer_global_mode=_closer_global_mode(f_subset, Fn_global, i))
            else:
                # Traceability only: nearest available subset mode, outside the tolerance
                row.update(pair_status="rejected_freq_tol", rejected_freq_tol=True)
        mode_rows.append(row)

    for j in unassigned:
        mode_rows.append({"pair_status": "unmatched_subset_mode", "subset_mode": f"Subset_Mode_{j + 1}",
                          "f_subset_Hz": float(Fn_subset_match[j]), "reliable": False,
                          "rejected_freq_tol": False})
    return mode_rows, len(Fn_subset_match), None


def compute_subset_metrics(mode_rows, n_global):
    """Metrics over the matched pairs only (rejected and unmatched modes excluded)."""
    matched = [r for r in mode_rows if r["pair_status"] == "matched"]
    n_matched = len(matched)
    n_reliable = int(sum(r["reliable"] for r in matched))
    abs_df = np.array([r["abs_df_pct"] for r in matched], dtype=float)
    macs = np.array([r["MAC"] for r in matched], dtype=float)
    return {
        "n_matched": n_matched,
        "n_rejected_freq_tol": sum(r["pair_status"] == "rejected_freq_tol" for r in mode_rows),
        "n_unmatched_subset_modes": sum(r["pair_status"] == "unmatched_subset_mode" for r in mode_rows),
        "n_reliable": n_reliable,
        "frac_reliable": n_reliable / n_global if n_global > 0 else np.nan,
        "mean_abs_df_pct": abs_df.mean() if n_matched else np.nan,
        "max_abs_df_pct": abs_df.max() if n_matched else np.nan,
        "MAC_min": macs.min() if n_matched else np.nan,
        "MAC_mean": macs.mean() if n_matched else np.nan,
    }


# ============================================================
# ONE RECORD (from the stored modal results)
# ============================================================
def process_record(record, modal, role, training_modes=None):
    """
    Match the stored subset modes of one record to its stored global modes.
    training_modes: (Fn, Phi) of the training global OMA for the optional tracking;
    None for the training record itself (its own global modes are used).
    """
    out = {"global_ok": False, "Fn_global": None, "Phi_global": None,
           "subset_rows": [], "mode_rows": [], "tracking_rows": []}

    if "global" not in modal:
        for subset_id, subset in enumerate(v1.SUBSETS, start=1):
            out["subset_rows"].append({"role": role, "record": record, **v1.subset_meta(subset_id, subset),
                                       "status": "global_oma_failed", "error": "no global OMA result in v1"})
        logger.error(f"[{record}] no global OMA result in {MODAL_RESULTS_FILE.name}")
        return out

    Fn_global = np.ravel(np.asarray(modal["global"]["Fn_Hz"], dtype=float))
    Phi_global = _as_mode_matrix(modal["global"]["Phi"], len(v1.CHANNELS))
    n_global = len(Fn_global)
    out.update(global_ok=True, Fn_global=Fn_global, Phi_global=Phi_global)

    # OPTIONAL: tracking of the record's global modes to the training global modes (v1 function)
    tracking_lookup = {}
    if v1.DO_MODE_TRACKING:
        Fn_ref, Phi_ref = training_modes if training_modes is not None else (Fn_global, Phi_global)
        for row in v1.track_to_training_modes(Fn_global, Phi_global, Fn_ref, Phi_ref):
            out["tracking_rows"].append({"role": role, "record": record, **row})
            tracking_lookup[row["global_mode"]] = row

    for subset_id, subset in enumerate(v1.SUBSETS, start=1):
        meta = v1.subset_meta(subset_id, subset)
        row = {"role": role, "record": record, **meta, "n_global_modes": n_global}

        if meta["subset_name"] not in modal:
            row.update(status="oma_failed", error="no subset OMA result in v1")
            out["subset_rows"].append(row)
            continue

        Fn_subset = np.ravel(np.asarray(modal[meta["subset_name"]]["Fn_Hz"], dtype=float))
        Phi_subset = _as_mode_matrix(modal[meta["subset_name"]]["Phi"], len(subset))
        mode_rows, n_prefiltered, error = match_subset_to_global(
            Fn_subset, Phi_subset, Fn_global, Phi_global, subset)
        row.update(n_subset_modes=len(Fn_subset), n_subset_modes_prefiltered=n_prefiltered)

        if error is not None:
            logger.warning(f"[{record}] {meta['subset_name']}: matching_failed: {error}")
            row.update(status="matching_failed", error=error)
            out["subset_rows"].append(row)
            continue

        row.update(status="ok", error="", **compute_subset_metrics(mode_rows, n_global))
        out["subset_rows"].append(row)

        for mode_row in mode_rows:
            tracked = tracking_lookup.get(mode_row.get("global_mode"), {})
            out["mode_rows"].append({
                "role": role, "record": record, **meta, **mode_row,
                "training_mode": tracked.get("training_mode"),
                "f_training_Hz": tracked.get("f_training_Hz", np.nan),
                "tracking_MAC": tracked.get("tracking_MAC", np.nan),
            })

        logger.info(f"[{record}] {meta['subset_name']} ({meta['subset_sensors']}): "
                    f"{n_global} global / {n_prefiltered} subset modes -> {row['n_matched']} matched, "
                    f"{row['n_reliable']} reliable, {row['n_rejected_freq_tol']} global without candidate, "
                    f"{row['n_unmatched_subset_modes']} extra subset; MAC_min = {row['MAC_min']:.3f}, "
                    f"max|df| = {row['max_abs_df_pct']:.2f} %")
    return out


# ============================================================
# TABLE 6 CHECK (training record)
# ============================================================
def check_table6_pairings(training_mode_rows, Fn_global_training):
    rows = []
    for subset_id, subset in enumerate(v1.SUBSETS, start=1):
        meta = v1.subset_meta(subset_id, subset)
        expected = TABLE6_SUBSET_FREQS_HZ.get(meta["subset_name"])
        if expected is None:
            continue
        obtained = {r["global_mode"]: r for r in training_mode_rows
                    if r["subset_name"] == meta["subset_name"] and r["pair_status"] == "matched"}
        for k, f_expected in enumerate(expected, start=1):
            pair = obtained.get(f"Mode_{k}")
            f_obtained = pair["f_subset_Hz"] if pair else np.nan
            if f_expected is None:
                same = pair is None
            else:
                same = pair is not None and abs(f_obtained - f_expected) <= TABLE6_FREQ_TOL_HZ
            rows.append({
                **meta,
                "gating": meta["subset_name"] in TABLE6_GATING_SUBSETS,
                "global_mode": f"Mode_{k}",
                "f_global_Hz": float(Fn_global_training[k - 1]) if k <= len(Fn_global_training) else np.nan,
                "f_subset_table6_Hz": np.nan if f_expected is None else f_expected,
                "f_subset_obtained_Hz": f_obtained,
                "MAC_obtained": pair["MAC"] if pair else np.nan,
                "same_pairing": bool(same),
            })
    df = pd.DataFrame(rows)
    passed = bool(df.loc[df["gating"], "same_pairing"].all())
    return df, passed


# ============================================================
# AGGREGATION AND RANKING (test records only)
# ============================================================
def rank_subsets(df_test):
    """
    Ranking over the test records where ALL subsets have status ok and finite values
    of every ranking criterion. A unique best counts as a win; a shared best as a tie.
    """
    finite = np.ones(len(df_test), dtype=bool)
    for crit in v1.RANKING_CRITERIA:
        finite &= np.isfinite(df_test[crit].astype(float)).to_numpy()
    valid = df_test[(df_test["status"] == "ok").to_numpy() & finite]
    complete_records = [record for record, grp in valid.groupby("record")
                        if grp["subset_name"].nunique() == len(v1.SUBSETS)]
    complete_records = sorted(complete_records, key=lambda r: v1._record_number(pathlib.Path(r)))

    counts = {}
    for subset_id, subset in enumerate(v1.SUBSETS, start=1):
        name = v1.subset_meta(subset_id, subset)["subset_name"]
        counts[name] = {}
        for crit in v1.RANKING_CRITERIA:
            counts[name].update({f"rank_{crit}_wins": 0, f"rank_{crit}_ties": 0})

    for record in complete_records:
        grp = valid[valid["record"] == record]
        for crit in v1.RANKING_CRITERIA:
            values = grp[crit].to_numpy(dtype=float)
            best = np.isclose(values, values.max(), rtol=0.0, atol=v1.TIE_ATOL)
            kind = "wins" if best.sum() == 1 else "ties"
            for name, is_best in zip(grp["subset_name"], best):
                if is_best:
                    counts[name][f"rank_{crit}_{kind}"] += 1

    df_counts = pd.DataFrame.from_dict(counts, orient="index").rename_axis("subset_name").reset_index()
    df_counts.insert(1, "n_records_ranked", len(complete_records))
    return df_counts, complete_records


def aggregate(df_test):
    rows = []
    for subset_id, subset in enumerate(v1.SUBSETS, start=1):
        meta = v1.subset_meta(subset_id, subset)
        grp = df_test[df_test["subset_name"] == meta["subset_name"]]
        row = {
            **meta,
            "n_records": grp["record"].nunique(),
            "n_ok": int((grp["status"] == "ok").sum()),
            "n_matching_failed": int((grp["status"] == "matching_failed").sum()),
            "n_oma_failed": int(grp["status"].isin(["oma_failed", "global_oma_failed"]).sum()),
        }
        for metric in AGGREGATED_METRICS:
            values = grp[metric].astype(float).dropna().to_numpy()
            if values.size:
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                stats = {"median": median, "min": values.min(), "max": values.max(), "Q1": q1, "Q3": q3}
            else:
                stats = dict.fromkeys(["median", "min", "max", "Q1", "Q3"], np.nan)
            row[f"{metric}_n"] = int(values.size)
            row.update({f"{metric}_{stat}": value for stat, value in stats.items()})
        rows.append(row)

    df_ranking, complete_records = rank_subsets(df_test)
    return pd.DataFrame(rows).merge(df_ranking, on="subset_name", how="left"), complete_records


def write_tables(subset_rows, mode_rows, tracking_rows):
    df_subsets = pd.DataFrame(subset_rows, columns=SUBSET_ROW_COLUMNS)
    df_modes = pd.DataFrame(mode_rows, columns=MODE_ROW_COLUMNS)
    df_subsets.to_csv(OUTPUT_DIR / "per_record_subset_metrics.csv", index=False)
    df_modes.to_csv(OUTPUT_DIR / "per_mode_matching.csv", index=False)
    if v1.DO_MODE_TRACKING:
        pd.DataFrame(tracking_rows, columns=v1.TRACKING_ROW_COLUMNS).to_csv(
            OUTPUT_DIR / "global_mode_tracking.csv", index=False)
    return df_subsets, df_modes


# ============================================================
# MAIN
# ============================================================
def main():
    setup_output_and_logging()

    with open(V1_DIR / "settings.json", encoding="utf-8") as f:
        v1_settings = json.load(f)
    modal_results = outils.load_json_serialized(MODAL_RESULTS_FILE)
    write_settings(v1_settings, v1.sha256_of_file(MODAL_RESULTS_FILE))

    training_record = pathlib.Path(v1_settings["training_file"]["file"]).stem
    test_records = sorted((r for r in modal_results if r != training_record),
                          key=lambda r: v1._record_number(pathlib.Path(r)))
    n_test_files = len(v1_settings["test_files"])

    # Exclusion is a data-level check, unchanged from v1
    shutil.copyfile(V1_DIR / "excluded_records.csv", OUTPUT_DIR / "excluded_records.csv")
    excluded = pd.read_csv(OUTPUT_DIR / "excluded_records.csv")

    logger.info(f"Modal results: {MODAL_RESULTS_FILE}")
    logger.info(f"Matching: frequency-gated greedy, global modes first, FREQ_TOL_PCT = {FREQ_TOL_PCT} %")

    # ------------------------------------------------------------
    # 1. Training record: frequency check and Table 6 pairing check
    # ------------------------------------------------------------
    logger.info("=== TRAINING RECORD ===")
    training_result = process_record(training_record, modal_results[training_record], "training")
    if not training_result["global_ok"]:
        raise RuntimeError("No global OMA result for the training record")

    df_sanity, sanity_passed = v1.sanity_check_training(training_result["Fn_global"])
    df_sanity.to_csv(OUTPUT_DIR / "sanity_check_training.csv", index=False)
    if not sanity_passed:
        logger.error("Training global frequencies do not match the expected values:\n"
                     + df_sanity.to_string(index=False))
        raise SystemExit(1)
    logger.info("Training global frequencies: check passed.")

    df_table6, table6_passed = check_table6_pairings(training_result["mode_rows"], training_result["Fn_global"])
    df_table6.to_csv(OUTPUT_DIR / "table6_pairing_check.csv", index=False)
    logger.info("Training pairings vs Table 6:\n" + df_table6.drop(columns=["subset_id", "subset_name"])
                .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    subset_rows = list(training_result["subset_rows"])
    mode_rows = list(training_result["mode_rows"])
    tracking_rows = list(training_result["tracking_rows"])
    write_tables(subset_rows, mode_rows, tracking_rows)

    if not table6_passed:
        differences = df_table6[df_table6["gating"] & ~df_table6["same_pairing"]]
        logger.error("TABLE 6 CHECK FAILED: the gated rule does not reproduce the Table 6 pairings of "
                     f"{TABLE6_GATING_SUBSETS}. Differences:\n" + differences.to_string(index=False)
                     + "\nTest records not processed.")
        raise SystemExit(1)
    logger.info(f"Table 6 check passed for {TABLE6_GATING_SUBSETS}.")
    training_modes = (training_result["Fn_global"], training_result["Phi_global"])

    # ------------------------------------------------------------
    # 2. Test records
    # ------------------------------------------------------------
    logger.info("=== TEST RECORDS ===")
    global_failures = []
    for record in test_records:
        result = process_record(record, modal_results[record], "test", training_modes)
        if not result["global_ok"]:
            global_failures.append(record)
        subset_rows.extend(result["subset_rows"])
        mode_rows.extend(result["mode_rows"])
        tracking_rows.extend(result["tracking_rows"])

    df_subsets, df_modes = write_tables(subset_rows, mode_rows, tracking_rows)

    # ------------------------------------------------------------
    # 3. Aggregation, ranking and (optional) mode recovery counts
    # ------------------------------------------------------------
    df_test = df_subsets[df_subsets["role"] == "test"]
    df_modes_test = df_modes[df_modes["role"] == "test"]
    processed_records = [r for r in test_records if r not in global_failures]

    df_aggregated, complete_records = aggregate(df_test)
    df_aggregated.to_csv(OUTPUT_DIR / "aggregated_metrics.csv", index=False)

    if v1.DO_MODE_TRACKING:
        df_tracking = pd.DataFrame(tracking_rows, columns=v1.TRACKING_ROW_COLUMNS)
        df_recovery = v1.mode_recovery_counts(
            df_modes_test, df_tracking[df_tracking["role"] == "test"], len(processed_records), training_modes[0])
        df_recovery.to_csv(OUTPUT_DIR / "mode_recovery_counts.csv", index=False)

    # ------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------
    subset_oma_failures = df_test[df_test["status"] == "oma_failed"]
    matching_failures = df_test[df_test["status"] == "matching_failed"]
    matched_test = df_modes_test[df_modes_test["pair_status"] == "matched"]
    wide_pairs = matched_test[matched_test["abs_df_pct"] > DF_DIAGNOSTIC_PCT]
    closer_pairs = matched_test[matched_test["closer_global_mode"].fillna("") != ""]

    lines = ["", "=" * 60, "SUMMARY (v2)", "=" * 60,
             f"Test records found:     {n_test_files}",
             f"Excluded:               {len(excluded)}"]
    lines += [f"    - {r.file}: {r.reason}" for r in excluded.itertuples()]
    lines += [f"Processed:              {len(processed_records)} ({', '.join(processed_records)})",
              f"Global OMA failures:    {len(global_failures)}",
              f"Subset OMA failures:    {len(subset_oma_failures)}",
              f"Matching failures:      {len(matching_failures)}"]
    lines += [f"    - {r.record} | {r.subset_name}: {r.error}" for r in matching_failures.itertuples()]
    lines += [f"Records ranked (all three subsets valid): {len(complete_records)} "
              f"({', '.join(complete_records)})",
              f"Matched test pairs: {len(matched_test)}; with {DF_DIAGNOSTIC_PCT} % < |df| <= {FREQ_TOL_PCT} %: "
              f"{len(wide_pairs)}; with a different global mode closer in frequency: {len(closer_pairs)}"]
    lines += ["", "Per-subset medians over the test records [Q1, Q3]:"]
    for r in df_aggregated.itertuples():
        lines.append(
            f"    {r.subset_name} ({r.subset_sensors}): "
            f"n_reliable = {r.n_reliable_median:.1f} [{r.n_reliable_Q1:.1f}, {r.n_reliable_Q3:.1f}], "
            f"frac_reliable = {r.frac_reliable_median:.3f} [{r.frac_reliable_Q1:.3f}, {r.frac_reliable_Q3:.3f}], "
            f"MAC_min = {r.MAC_min_median:.3f} [{r.MAC_min_Q1:.3f}, {r.MAC_min_Q3:.3f}], "
            f"max|df| = {r.max_abs_df_pct_median:.2f} %"
        )
        lines.append(
            f"        ranking over {r.n_records_ranked} records: best MAC_min {r.rank_MAC_min_wins} wins + "
            f"{r.rank_MAC_min_ties} ties; best frac_reliable {r.rank_frac_reliable_wins} wins + "
            f"{r.rank_frac_reliable_ties} ties"
        )
    lines.append(f"\nOutputs written to: {OUTPUT_DIR}")
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
