"""
All 3-sensor layouts (120): out-of-sample metrics vs the objective J - Mendez-Nunez bridge.

Light, re-matching only (no OMA is run). Inputs:
    results/oos_bridge/all_layouts/modes/<record>.jsonl   (OMA_Code_1_OOS_4_all_layouts_run.py)
    results/oos_bridge/modal_results.json                 (OMA_Code_1_OOS_1_run_oma.py, global modes)
    results/reference_modes/reference_modes.json          (OMA_Code_1_OOS_2_reference_modes.py)
    results/oos_bridge/v3/per_record_subset_metrics.csv   (OMA_Code_1_OOS_3_validate_subsets.py, sanity check)
    training record, signal_Geo.xlsx, results/EFDD_results.json (objective J, as brute_force_sensor_selection.py)

Matching and metrics are those of OMA_Code_1_OOS_3_validate_subsets.py (imported): Ref labelling
of the global modes, frequency-gated greedy matcher on the Ref-labelled global modes, reliable
pair MAC >= 0.8, versions "all_ref" and "robust". Identification failure convention: a failed
subset OMA, or a version with no matched pair, gets frac_reliable = MAC_min = MAC_mean = 0 and
|df| = NaN (left out of the |df| statistics); both cases are counted separately.

The objective J of the submitted paper is recomputed for all 120 layouts with the functions
and settings of brute_force_sensor_selection.py / qaoa-fm_OMA.py (imported, not modified) and
checked against bruteforce_results/bruteforce_ranking_all_combinations.csv.

Primary metrics (fixed before looking at the results), version "robust", 8 test records:
    median of frac_reliable; worst case (minimum) of MAC_min.

Outputs: results/oos_bridge/all_layouts/analysis/
"""
import ast
import json
import logging
import pathlib
import platform
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # batch run: no interactive figure windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import helpers.outils as outils
import OMA_Code_1_OOS_1_run_oma as v1                # frozen settings and shared helpers
import OMA_Code_1_OOS_3_validate_subsets as v3       # Ref labelling, matcher, metrics
import OMA_Code_1_OOS_4_all_layouts_run as run4      # layouts and the per-record mode store
import brute_force_sensor_selection as bf            # objective J: settings and term decomposition (sets the paper plot style)

# ============================================================
# SETTINGS
# ============================================================
VERSIONS = v3.VERSIONS                  # ["all_ref", "robust"]
METRICS = v3.METRICS                    # n_ref_present ... MAC_mean
OOS_STATS = ["median", "Q1", "Q3", "min"]
EQUALITY_ATOL = 1e-9                    # sanity checks (v3 metrics, saved J)
TIE_ATOL = 1e-12                        # values closer than this count as tied
TOP_K = 10
ARGMIN_J_EXPECTED = [1, 2, 5]           # {S2,S3,S6}

# Layout-level metrics: (column, role, higher_is_better). Primary metrics fixed before the analysis.
LAYOUT_METRICS = [
    ("OOS_robust_frac_reliable_median", "primary", True),
    ("OOS_robust_MAC_min_min", "primary", True),
    ("OOS_all_ref_frac_reliable_median", "secondary", True),
    ("OOS_all_ref_MAC_min_min", "secondary", True),
    ("OOS_robust_MAC_mean_median", "secondary", True),
    ("OOS_all_ref_MAC_mean_median", "secondary", True),
    ("OOS_robust_mean_abs_df_pct_median", "secondary", False),
    ("OOS_all_ref_mean_abs_df_pct_median", "secondary", False),
    ("IS_robust_frac_reliable", "secondary", True),
    ("IS_robust_MAC_min", "secondary", True),
    ("IS_all_ref_frac_reliable", "secondary", True),
    ("IS_all_ref_MAC_min", "secondary", True),
]
OOS_METRIC_COLUMNS = [m for m in LAYOUT_METRICS if m[0].startswith("OOS_")]
# In-sample vs out-of-sample pairs for the correlations
IS_OOS_PAIRS = [(f"IS_{v}_{m}", f"OOS_{v}_{m}_{s}")
                for v in VERSIONS
                for m, s in [("frac_reliable", "median"), ("frac_reliable", "min"),
                             ("MAC_min", "median"), ("MAC_min", "min"),
                             ("MAC_mean", "median"), ("mean_abs_df_pct", "median")]]

V3_METRICS_FILE = v1.OUTPUT_DIR / "v3" / "per_record_subset_metrics.csv"
BRUTEFORCE_RANKING_FILE = bf.OUT_DIR / "bruteforce_ranking_all_combinations.csv"
OUTPUT_DIR = run4.OUTPUT_DIR / "analysis"

# Figure colours (validated reference palette, light mode; first three slots pass all pairs)
COLOR_ALL = "#a3a29c"
HIGHLIGHT_STYLE = {  # Table-6 layouts; {S2,S3,S6} is also argmin J
    "subset_1_2_5": {"color": "#2a78d6", "marker": "*", "size": 220},
    "subset_0_1_7": {"color": "#eb6834", "marker": "D", "size": 70},
    "subset_1_2_7": {"color": "#1baf7a", "marker": "s", "size": 70},
}
TEXT_COLOR = "#0b0b0b"
GRID_COLOR = "#e4e3df"

RECORD_ROW_COLUMNS = [
    "role", "record", "layout_id", "layout_name", "layout_sensors", "version", "status",
    "identification_failure", "n_global_modes", "n_nonref_global_modes", "n_subset_modes",
    "n_unmatched_subset_modes", *METRICS,
]

logger = logging.getLogger("oos_bridge_all_layouts_analysis")


# ============================================================
# SET-UP
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


def write_settings(records, j_info):
    inputs = {
        "modal_results": v3.MODAL_RESULTS_FILE,
        "v1_settings": v3.V1_SETTINGS_FILE,
        "reference_modes": v3.REFERENCE_MODES_FILE,
        "consistency_MAC_assigned": v3.CONSISTENCY_FILE,
        "v3_per_record_subset_metrics": V3_METRICS_FILE,
        "all_layouts_settings": run4.OUTPUT_DIR / "settings.json",
        **{f"store_{r}": run4.store_file(r) for r in records},
        "training_record_for_J": bf.EXCEL_FILE,
        "geometry_for_J": bf.GEOMETRY_FILE,
        "EFDD_results_for_J": bf.OMA_JSON_FILE,
        "bruteforce_ranking": BRUTEFORCE_RANKING_FILE,
    }
    settings = {
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matching_and_metrics": "imported from OMA_Code_1_OOS_3_validate_subsets.py",
        "REF_LABEL_MAC_THRESHOLD": v3.REF_LABEL_MAC_THRESHOLD,
        "FREQ_TOL_PCT": v3.FREQ_TOL_PCT,
        "MAC_THRESHOLD": v3.MAC_THRESHOLD,
        "ROBUST_REF_LABELS": v3.ROBUST_REF_LABELS,
        "VERSIONS": VERSIONS,
        "identification_failure": "failed subset OMA, or no matched pair in a version: frac_reliable = "
                                  "MAC_min = MAC_mean = 0, mean/max |df| = NaN (left out of the |df| "
                                  "statistics); counted separately as oma_failed / no_match",
        "primary_metrics": ["OOS_robust_frac_reliable_median", "OOS_robust_MAC_min_min"],
        "layout_metrics": [{"column": c, "role": r, "higher_is_better": h} for c, r, h in LAYOUT_METRICS],
        "OOS_STATS": OOS_STATS,
        "rank": "1 = best; higher is better for frac_reliable and MAC metrics, lower is better for J and "
                "|df|; average ranks for ties; NaN values are not ranked",
        "percentile": "100 * (n - rank) / (n - 1): 100 = best layout, 0 = worst",
        "correlation": "Spearman rank correlation (scipy.stats.spearmanr), pairs with NaN dropped",
        "EQUALITY_ATOL": EQUALITY_ATOL,
        "TIE_ATOL": TIE_ATOL,
        "TOP_K": TOP_K,
        "objective_J": j_info,
        "inputs": {name: {"path": str(path), "sha256": v1.sha256_of_file(path)} for name, path in inputs.items()},
    }
    with open(OUTPUT_DIR / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ============================================================
# OBJECTIVE J (brute_force_sensor_selection.py, recomputed)
# ============================================================
def compute_objective():
    """
    J and its terms for the 120 layouts, with exactly the settings and functions of
    brute_force_sensor_selection.py (main, steps 1-4), plus J without the FIM utility term.
    """
    qaoa = bf.load_main_module(bf.MAIN_QAOA_FILE)
    df = qaoa.load_and_prep_data(str(bf.EXCEL_FILE))
    sensor_names = df.columns.tolist()
    if sensor_names != v1.CHANNELS:
        raise SystemExit(f"Sensor order of {bf.EXCEL_FILE.name} differs from v1.CHANNELS: {sensor_names}")

    def utility_and_redundancy(w_fim):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # w_fim = 0 renormalises the utility weights (b is min-max scaled)
            return qaoa.compute_metrics_from_data(
                df=df, sample_rate_hz=bf.SAMPLE_RATE_HZ, w_entropy=bf.W_ENTROPY, w_spectral=bf.W_SPECTRAL,
                w_fim=w_fim, n_bands=bf.N_SPECTRAL_BANDS, w_mi=bf.W_MI, w_band_corr=bf.W_BAND_CORR,
                oma_json_path=str(bf.OMA_JSON_FILE))

    b, Corr, diagnostics = utility_and_redundancy(bf.W_FIM)
    b_no_fim, _, _ = utility_and_redundancy(0.0)

    geom_df = qaoa.load_sensor_geometry(str(bf.GEOMETRY_FILE))
    prox = qaoa.build_proximity_matrix(qaoa.build_distance_matrix(sensor_names, geom_df),
                                       length_scale=bf.GEOM_LENGTH_SCALE)
    left_idx, right_idx, axis = qaoa.infer_left_right_groups(sensor_names, geom_df, symmetry_coord=bf.SYMMETRY_COORD)

    def qubo(b_vec):
        return qaoa.build_qubo_sensor_selection(
            b=b_vec, Corr=Corr, k=bf.K_SENSORS_TO_SELECT, prox_matrix=prox, left_indices=left_idx,
            right_indices=right_idx, lam=bf.QUBO_LAMBDA, alpha=bf.QUBO_ALPHA, gamma=bf.GEOM_GAMMA, eta_sym=bf.SYM_ETA)

    Q, Q_no_fim = qubo(b), qubo(b_no_fim)
    offset = bf.QUBO_LAMBDA * bf.K_SENSORS_TO_SELECT ** 2

    rows = []
    for layout in run4.LAYOUTS:
        terms = bf.decompose_subset_terms(
            selected_indices=layout, b=b, Corr=Corr, prox_matrix=prox, left_indices=left_idx,
            right_indices=right_idx, alpha=bf.QUBO_ALPHA, gamma=bf.GEOM_GAMMA, eta_sym=bf.SYM_ETA,
            lam=bf.QUBO_LAMBDA, k=bf.K_SENSORS_TO_SELECT)
        J = bf.subset_energy_from_Q(Q, layout, constant_offset=offset)
        J_no_fim = bf.subset_energy_from_Q(Q_no_fim, layout, constant_offset=offset)
        rows.append({
            "layout_name": run4.layout_meta(layout)["subset_name"],
            "J": J,
            "J_utility_term": terms["utility_term"],
            "J_redundancy_term": terms["redundancy_term"],
            "J_geometry_term": terms["geometry_term"],
            "J_symmetry_term": terms["symmetry_term"],
            "J_cardinality_term": terms["cardinality_term"],
            "J_from_terms": terms["total_energy_from_terms"],
            "J_without_FIM": J_no_fim,
            "J_FIM_effect": J - J_no_fim,
        })
    df_J = pd.DataFrame(rows)
    info = {
        "source": "brute_force_sensor_selection.py (settings, decompose_subset_terms, subset_energy_from_Q) "
                  "with qaoa-fm_OMA.py (descriptors, QUBO), both imported unchanged",
        "definition": "J = x^T Q x + lambda * k^2 = utility + redundancy + geometry + symmetry + cardinality",
        "weights": {"W_ENTROPY": bf.W_ENTROPY, "W_SPECTRAL": bf.W_SPECTRAL, "W_FIM": bf.W_FIM, "W_MI": bf.W_MI,
                    "W_BAND_CORR": bf.W_BAND_CORR, "N_SPECTRAL_BANDS": bf.N_SPECTRAL_BANDS,
                    "GEOM_LENGTH_SCALE": bf.GEOM_LENGTH_SCALE, "GEOM_GAMMA": bf.GEOM_GAMMA, "SYM_ETA": bf.SYM_ETA,
                    "QUBO_LAMBDA": bf.QUBO_LAMBDA, "QUBO_ALPHA": bf.QUBO_ALPHA, "K": bf.K_SENSORS_TO_SELECT},
        "modal_data": "only the Ed vector of results/EFDD_results.json (FIM utility term)",
        "Ed": [float(x) for x in diagnostics["Ed_raw"]],
        "b_fim": [float(x) for x in diagnostics["b_fim"]],
        "symmetry_axis_x": float(axis),
        "J_without_FIM": "same J with w_fim = 0 (utility weights renormalised; b is min-max scaled)",
    }
    return df_J, info


def verify_objective(df_J):
    """J and its terms must reproduce the saved brute-force ranking; argmin J must be {S2,S3,S6}."""
    saved = pd.read_csv(BRUTEFORCE_RANKING_FILE)
    saved["layout_name"] = saved["subset_indices"].map(
        lambda s: run4.layout_meta(ast.literal_eval(s))["subset_name"])
    merged = df_J.merge(saved, on="layout_name", how="outer", indicator=True)
    pairs = [("J", "energy"), ("J_utility_term", "utility_term"), ("J_redundancy_term", "redundancy_term"),
             ("J_geometry_term", "geometry_term"), ("J_symmetry_term", "symmetry_term"),
             ("J_cardinality_term", "cardinality_term"), ("J_from_terms", "total_energy_from_terms")]
    problems = []
    if not (merged["_merge"] == "both").all() or len(merged) != len(run4.LAYOUTS):
        problems.append("layout sets differ")
    for new, old in pairs:
        diff = np.abs(merged[new] - merged[old]).max()
        if not diff <= EQUALITY_ATOL:
            problems.append(f"{new} vs {old}: max |diff| = {diff:.3e}")
    layout_by_name = {run4.layout_meta(layout)["subset_name"]: layout for layout in run4.LAYOUTS}
    argmin = layout_by_name[df_J.loc[df_J["J"].idxmin(), "layout_name"]]
    return problems, argmin, float(np.abs(merged["J"] - merged["energy"]).max())


# ============================================================
# MATCHING AND METRICS (OOS_3 functions)
# ============================================================
def label_record(modal_record, ref):
    Fn_global = np.ravel(np.asarray(modal_record["global"]["Fn_Hz"], dtype=float))
    Phi_global = v3._as_mode_matrix(modal_record["global"]["Phi"], len(v1.CHANNELS))
    label_rows, labels = v3.label_global_modes(Fn_global, Phi_global, ref)
    ref_idx = [i for i, label in enumerate(labels) if label is not None]
    return {"Fn": Fn_global, "Phi": Phi_global, "labels": labels, "ref_idx": ref_idx,
            "n_nonref": len(labels) - len(ref_idx),
            "n_present": {"all_ref": len(ref_idx),
                          "robust": sum(labels[i] in v3.ROBUST_REF_LABELS for i in ref_idx)}}


def pair_rows(assignments, glob, Fn_subset_match):
    """One row per Ref-labelled global mode, as in v3.process_record."""
    rows = []
    for assignment, i in zip(assignments, glob["ref_idx"]):
        label = glob["labels"][i]
        f_global = float(glob["Fn"][i])
        row = {"ref_label": label, "robust": label in v3.ROBUST_REF_LABELS, "f_global_Hz": f_global,
               "pair_status": "no_subset_mode_left", "reliable": False}
        j = assignment["ref_index"]
        if j is not None:
            df_pct = 100.0 * (float(Fn_subset_match[j]) - f_global) / f_global
            row.update(f_subset_Hz=float(Fn_subset_match[j]), df_pct=df_pct, abs_df_pct=abs(df_pct),
                       MAC=assignment["MAC"])
            if assignment["matched"]:
                if abs(df_pct) > v3.FREQ_TOL_PCT + 1e-9:
                    raise RuntimeError("matched pair outside the frequency gate")
                row.update(pair_status="matched", reliable=bool(assignment["MAC"] >= v3.MAC_THRESHOLD))
            else:
                row["pair_status"] = "rejected_freq_tol"
        rows.append(row)
    return rows


def failure_metrics(n_present):
    """Identification failure convention."""
    return {"n_ref_present": n_present, "n_matched": 0, "n_reliable": 0, "frac_reliable": 0.0,
            "mean_abs_df_pct": np.nan, "max_abs_df_pct": np.nan, "MAC_min": 0.0, "MAC_mean": 0.0}


def layout_record_rows(record, role, layout, entry, glob):
    meta = run4.layout_meta(layout)
    base = {"role": role, "record": record, "layout_id": meta["subset_id"], "layout_name": meta["subset_name"],
            "layout_sensors": meta["subset_sensors"], "n_global_modes": len(glob["Fn"]),
            "n_nonref_global_modes": glob["n_nonref"]}

    if entry["status"] != "ok":
        return [{**base, "version": v, "status": "oma_failed", "identification_failure": True,
                 **failure_metrics(glob["n_present"][v])} for v in VERSIONS]

    Fn_subset = np.ravel(np.asarray(entry["Fn_Hz"], dtype=float))
    Phi_subset = v3._as_mode_matrix(entry["Phi"], len(layout))
    try:
        assignments, Fn_subset_match, unassigned = v3.match_subset_to_reference(
            Fn_subset, Phi_subset, glob["Fn"][glob["ref_idx"]], glob["Phi"][:, glob["ref_idx"]], layout)
    except Exception as exc:
        logger.warning(f"[{record}] {meta['subset_name']}: matching failed: {type(exc).__name__}: {exc}")
        return [{**base, "version": v, "status": "matching_failed", "identification_failure": True,
                 **failure_metrics(glob["n_present"][v])} for v in VERSIONS]

    rows_all = pair_rows(assignments, glob, Fn_subset_match)
    out = []
    for version in VERSIONS:
        rows_v = rows_all if version == "all_ref" else [r for r in rows_all if r["robust"]]
        metrics = v3.compute_metrics(rows_v)
        status = "ok" if metrics["n_matched"] > 0 else "no_match"
        if status == "no_match":
            metrics = failure_metrics(metrics["n_ref_present"])
        out.append({**base, "version": version, "status": status, "identification_failure": status != "ok",
                    "n_subset_modes": len(Fn_subset), "n_unmatched_subset_modes": len(unassigned), **metrics})
    return out


def check_against_v3(df_records):
    """Table-6 layouts: per-record metrics must equal results/oos_bridge/v3/per_record_subset_metrics.csv."""
    v3_rows = pd.read_csv(V3_METRICS_FILE)
    new = df_records[df_records["layout_name"].isin(v3_rows["subset_name"].unique())].rename(
        columns={"layout_name": "subset_name"})
    merged = v3_rows.merge(new, on=["record", "subset_name", "version"], how="outer", suffixes=("_v3", ""),
                           indicator=True)
    problems = []
    if not (merged["_merge"] == "both").all():
        problems.append(f"{int((merged['_merge'] != 'both').sum())} rows present in only one table")
    for metric in METRICS:
        a, b = merged[f"{metric}_v3"].astype(float), merged[metric].astype(float)
        bad = ~np.isclose(a, b, rtol=0.0, atol=EQUALITY_ATOL, equal_nan=True)
        if bad.any():
            problems.append(f"{metric}: {int(bad.sum())} rows differ (max |diff| = {np.nanmax(np.abs(a - b)):.3e})")
    if not (merged["status_v3"] == merged["status"]).all():
        problems.append("status differs")
    return problems, len(merged)


# ============================================================
# LAYOUT SUMMARY, RANKS, CORRELATIONS
# ============================================================
def summarise_layouts(df_records, df_J):
    rows = []
    for layout in run4.LAYOUTS:
        meta = run4.layout_meta(layout)
        row = {"layout_id": meta["subset_id"], "layout_name": meta["subset_name"],
               "layout_sensors": meta["subset_sensors"], "layout_indices": str(layout),
               "is_table6": layout in run4.TABLE6_LAYOUTS}
        grp = df_records[df_records["layout_name"] == meta["subset_name"]]
        for version in VERSIONS:
            train = grp[(grp["role"] == "training") & (grp["version"] == version)]
            for metric in METRICS:
                row[f"IS_{version}_{metric}"] = float(train[metric].iloc[0]) if not train.empty else np.nan
            test = grp[(grp["role"] == "test") & (grp["version"] == version)]
            row[f"OOS_{version}_n_records"] = test["record"].nunique()
            row[f"OOS_{version}_n_oma_failed"] = int((test["status"] == "oma_failed").sum())
            row[f"OOS_{version}_n_no_match"] = int((test["status"] == "no_match").sum())
            for metric in METRICS:
                values = test[metric].astype(float).dropna().to_numpy()
                q1, median, q3 = np.percentile(values, [25, 50, 75]) if values.size else (np.nan,) * 3
                row.update({f"OOS_{version}_{metric}_median": median, f"OOS_{version}_{metric}_Q1": q1,
                            f"OOS_{version}_{metric}_Q3": q3,
                            f"OOS_{version}_{metric}_min": values.min() if values.size else np.nan})
        rows.append(row)
    df = pd.DataFrame(rows).merge(df_J, on="layout_name", how="left")
    df.insert(df.columns.get_loc("J"), "J_rank", rank_values(df["J"], higher_is_better=False))
    return df


def rank_values(values, higher_is_better):
    """Rank 1 = best; average ranks for ties (TIE_ATOL); NaN values are not ranked."""
    values = np.asarray(values, dtype=float)
    ranks = np.full(values.shape, np.nan)
    ok = np.isfinite(values)
    keyed = np.round((-values[ok] if higher_is_better else values[ok]) / TIE_ATOL) * TIE_ATOL
    ranks[ok] = stats.rankdata(keyed, method="average")
    return ranks


def table6_ranks(df_layouts):
    metric_defs = [("J", "objective", False)] + LAYOUT_METRICS
    rows = []
    for column, role, higher in metric_defs:
        values = df_layouts[column].astype(float)
        ranks = rank_values(values, higher)
        n = int(np.isfinite(values).sum())
        for idx in df_layouts.index[df_layouts["is_table6"]]:
            value = values[idx]
            rows.append({
                "layout_name": df_layouts.at[idx, "layout_name"],
                "layout_sensors": df_layouts.at[idx, "layout_sensors"],
                "is_argmin_J": df_layouts.at[idx, "J_rank"] == 1,
                "metric": column, "role": role,
                "direction": "higher is better" if higher else "lower is better",
                "value": value, "rank": ranks[idx], "n_layouts_ranked": n,
                "percentile": 100.0 * (n - ranks[idx]) / (n - 1) if np.isfinite(ranks[idx]) and n > 1 else np.nan,
                "n_layouts_tied": int(np.sum(np.isclose(values, value, rtol=0.0, atol=TIE_ATOL)))
                                  if np.isfinite(value) else 0,
            })
    return pd.DataFrame(rows)


def spearman(df, x, y):
    data = df[[x, y]].astype(float).dropna()
    if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return np.nan, np.nan, len(data)
    rho, p = stats.spearmanr(data[x], data[y])
    return float(rho), float(p), len(data)


def correlations(df_layouts):
    rows = []
    for column, role, higher in OOS_METRIC_COLUMNS:
        rho, p, n = spearman(df_layouts, "J", column)
        rows.append({"group": "J vs OOS", "x": "J", "y": column, "y_role": role, "rho": rho, "p_value": p, "n": n,
                     "sign_if_J_predicts_OOS": "negative" if higher else "positive"})
    for x, y in IS_OOS_PAIRS:
        rho, p, n = spearman(df_layouts, x, y)
        rows.append({"group": "in-sample vs OOS", "x": x, "y": y,
                     "y_role": next((r for c, r, _ in LAYOUT_METRICS if c == y), "secondary"),
                     "rho": rho, "p_value": p, "n": n, "sign_if_J_predicts_OOS": "positive (same metric)"})
    return pd.DataFrame(rows)


def topk_by_J(df_layouts):
    primary = [c for c, r, _ in LAYOUT_METRICS if r == "primary"]
    top = df_layouts.nsmallest(TOP_K, "J")[["J_rank", "layout_name", "layout_sensors", "J", *primary]].copy()
    top.insert(0, "row_type", "layout")
    summary = []
    for group, data in ((f"top{TOP_K}_by_J", top), ("all_120", df_layouts)):
        for stat in ("median", "min", "max"):
            row = {"row_type": f"{group}_{stat}"}
            for c in ["J", *primary]:
                row[c] = getattr(data[c].astype(float), stat)()
            summary.append(row)
    return pd.concat([top, pd.DataFrame(summary)], ignore_index=True)


# ============================================================
# FIGURES
# ============================================================
def scatter_figure(df_layouts, x, y, xlabel, ylabel, title, filepath):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(df_layouts[x], df_layouts[y], s=22, color=COLOR_ALL, alpha=0.75, linewidths=0,
               label="All 120 layouts", zorder=2)
    for name, style in HIGHLIGHT_STYLE.items():
        row = df_layouts[df_layouts["layout_name"] == name].iloc[0]
        sensors = "{" + row["layout_sensors"].replace("-", ",") + "}"
        label = f"{sensors} (argmin J)" if row["J_rank"] == 1 else sensors
        ax.scatter(row[x], row[y], s=style["size"], marker=style["marker"], color=style["color"],
                   edgecolors="white", linewidths=1.5, label=label, zorder=4)
        ax.annotate(sensors, (row[x], row[y]), textcoords="offset points", xytext=(7, 5),
                    fontsize=9, color=TEXT_COLOR, zorder=5)
    rho, p, n = spearman(df_layouts, x, y)
    ax.set_title(f"{title}\nSpearman rho = {rho:.2f} (p = {p:.2g}, n = {n})", fontsize=11, color=TEXT_COLOR)
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    setup_output_and_logging()

    with open(v3.V1_SETTINGS_FILE, encoding="utf-8") as f:
        v1_settings = json.load(f)
    modal_results = outils.load_json_serialized(v3.MODAL_RESULTS_FILE)
    training_record = pathlib.Path(v1_settings["training_file"]["file"]).stem
    test_records = sorted((r for r in modal_results if r != training_record),
                          key=lambda r: v1._record_number(pathlib.Path(r)))
    records = [training_record] + test_records

    # ------------------------------------------------------------
    # 1. Objective J (recomputed, verified against the saved brute-force ranking)
    # ------------------------------------------------------------
    df_J, j_info = compute_objective()
    write_settings(records, j_info)
    problems, argmin, max_diff = verify_objective(df_J)
    if problems:
        logger.error("Objective J does not reproduce the saved brute-force ranking:\n    " + "\n    ".join(problems))
        raise SystemExit(1)
    logger.info(f"Objective J reproduces {BRUTEFORCE_RANKING_FILE.name} (max |diff| = {max_diff:.2e})")
    if argmin != ARGMIN_J_EXPECTED:
        logger.error(f"argmin J = {argmin}, expected {ARGMIN_J_EXPECTED}")
        raise SystemExit(1)
    logger.info(f"argmin J = {argmin} = {run4.layout_meta(argmin)['subset_sensors']} "
                f"(J = {df_J['J'].min():.6f}); FIM effect on J: max |J - J_without_FIM| = "
                f"{df_J['J_FIM_effect'].abs().max():.2e} (Ed = {sorted(set(np.round(j_info['Ed'], 12)))})")

    # ------------------------------------------------------------
    # 2. Inputs of the matching: store completeness, robust labels, reference modes
    # ------------------------------------------------------------
    stores = {r: run4.read_store(r) for r in records}
    incomplete = {r: len(s) for r, s in stores.items() if len(s) < len(run4.LAYOUTS)}
    if incomplete:
        raise SystemExit(f"Incomplete layout store (run OMA_Code_1_OOS_4_all_layouts_run.py first): {incomplete}")
    if set(v3.robust_labels_from_consistency(test_records)) != set(v3.ROBUST_REF_LABELS):
        raise SystemExit("ROBUST_REF_LABELS do not match consistency_MAC_assigned.csv")
    ref = v3.load_reference_modes()

    # ------------------------------------------------------------
    # 3. Per (layout, record, version) metrics
    # ------------------------------------------------------------
    rows = []
    for record in records:
        role = "training" if record == training_record else "test"
        glob = label_record(modal_results[record], ref)
        for layout in run4.LAYOUTS:
            entry = stores[record][run4.layout_meta(layout)["subset_name"]]
            rows += layout_record_rows(record, role, layout, entry, glob)
    df_records = pd.DataFrame(rows, columns=RECORD_ROW_COLUMNS)
    df_records.to_csv(OUTPUT_DIR / "per_layout_record_metrics.csv", index=False)

    problems, n_checked = check_against_v3(df_records)
    if problems:
        logger.error("SANITY CHECK FAILED: Table-6 layouts differ from v3:\n    " + "\n    ".join(problems))
        raise SystemExit(1)
    logger.info(f"Sanity check passed: Table-6 layouts reproduce {V3_METRICS_FILE.name} ({n_checked} rows)")

    # ------------------------------------------------------------
    # 4. Layout summary, ranks, correlations, top-k
    # ------------------------------------------------------------
    df_layouts = summarise_layouts(df_records, df_J)
    df_layouts.to_csv(OUTPUT_DIR / "per_layout_summary.csv", index=False)
    df_ranks = table6_ranks(df_layouts)
    df_ranks.to_csv(OUTPUT_DIR / "table6_layouts_rank.csv", index=False)
    df_corr = correlations(df_layouts)
    df_corr.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
    df_top = topk_by_J(df_layouts)
    df_top.to_csv(OUTPUT_DIR / "topk_by_J.csv", index=False)

    scatter_figure(df_layouts, "J", "OOS_robust_frac_reliable_median", "Objective J (lower is better)",
                   "OOS median frac_reliable (robust)", "Objective J vs out-of-sample mode recovery",
                   OUTPUT_DIR / "scatter_J_vs_OOS_robust_frac_reliable_median.png")
    scatter_figure(df_layouts, "J", "OOS_robust_MAC_min_min", "Objective J (lower is better)",
                   "OOS worst-case MAC_min (robust)", "Objective J vs out-of-sample worst-case MAC_min",
                   OUTPUT_DIR / "scatter_J_vs_OOS_robust_MAC_min_worst.png")
    scatter_figure(df_layouts, "IS_robust_MAC_min", "OOS_robust_MAC_min_min", "In-sample MAC_min (robust, training)",
                   "OOS worst-case MAC_min (robust)", "In-sample vs out-of-sample MAC_min",
                   OUTPUT_DIR / "scatter_IS_vs_OOS_robust_MAC_min.png")

    # ------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------
    test_rows = df_records[df_records["role"] == "test"]
    lines = ["", "=" * 60, "SUMMARY (all 120 layouts)", "=" * 60,
             f"Records: training ({training_record}) + {len(test_records)} test; "
             f"{len(run4.LAYOUTS)} layouts; {len(df_records)} (layout, record, version) rows",
             f"Objective J: verified against {BRUTEFORCE_RANKING_FILE.name}; argmin J = "
             f"{run4.layout_meta(argmin)['subset_sensors']}; FIM term has no effect on J "
             f"(max |J - J_without_FIM| = {df_J['J_FIM_effect'].abs().max():.1e})"]
    lines.append("Identification failures over the test records (layout x record):")
    for version in VERSIONS:
        t = test_rows[test_rows["version"] == version]
        lines.append(f"    {version}: oma_failed = {int((t['status'] == 'oma_failed').sum())}, "
                     f"no_match = {int((t['status'] == 'no_match').sum())}, "
                     f"matching_failed = {int((t['status'] == 'matching_failed').sum())} (of {len(t)})")
    lines += ["", "Table-6 layouts: rank (1 = best) and percentile among the 120 layouts:"]
    for metric in ["J", "OOS_robust_frac_reliable_median", "OOS_robust_MAC_min_min",
                   "OOS_all_ref_frac_reliable_median", "OOS_all_ref_MAC_min_min"]:
        sub = df_ranks[df_ranks["metric"] == metric]
        lines.append(f"    {metric}: " + "; ".join(
            f"{r.layout_sensors} {r.value:.3f} (rank {r.rank:.1f}, percentile {r.percentile:.0f}, "
            f"{r.n_layouts_tied} tied)"
            for r in sub.itertuples()))
    lines += ["", "Spearman correlations (n = 120):"]
    for r in df_corr.itertuples():
        if r.y_role == "primary" or r.group == "in-sample vs OOS" and r.y.endswith(("frac_reliable_median", "MAC_min_min")):
            lines.append(f"    {r.x} vs {r.y}: rho = {r.rho:.3f}, p = {r.p_value:.2g} (n = {r.n}; "
                         f"sign if predictive: {r.sign_if_J_predicts_OOS})")
    top = df_top[df_top["row_type"] != "layout"].set_index("row_type")
    lines += ["", f"Top {TOP_K} layouts by J vs all 120 (median [min, max]):"]
    for c in ["OOS_robust_frac_reliable_median", "OOS_robust_MAC_min_min"]:
        lines.append(f"    {c}: top {TOP_K} {top.at[f'top{TOP_K}_by_J_median', c]:.3f} "
                     f"[{top.at[f'top{TOP_K}_by_J_min', c]:.3f}, {top.at[f'top{TOP_K}_by_J_max', c]:.3f}]; "
                     f"all 120 {top.at['all_120_median', c]:.3f} "
                     f"[{top.at['all_120_min', c]:.3f}, {top.at['all_120_max', c]:.3f}]")
    lines.append(f"\nOutputs written to: {OUTPUT_DIR}")
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
