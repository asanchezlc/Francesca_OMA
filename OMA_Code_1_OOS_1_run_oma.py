"""
Out-of-sample validation of the FROZEN sensor subsets - Mendez-Nunez bridge.

For every independent record (Accelerations_XX.xlsx):
    1. Exclusion check on the raw 10 channels (applied before any OMA).
    2. Global COV-SSI on the full 10-channel network of THAT record (reference).
    3. COV-SSI on each frozen 3-sensor subset.
    4. Matching of the subset modes to the global modes of the same record with
       helpers.outils.match_modes_fem_oma_II (same arguments as the base script).

The training record (Roberto_signal_Puente_Circunv) is run first as a sanity
check; its results are written with role = "training" and are left out of the
aggregation and the ranking.

Base script: OMA_Code_1_COVSSI_subsets.py (global run with opt = 1).
"""
import hashlib
import json
import logging
import pathlib
import platform
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # batch run: no interactive figure windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helpers.outils as outils

from pyoma2.algorithms.ssi import SSI
from pyoma2.algorithms.data.run_params import SSIRunParams, Clustering, Step1, Step2, Step3
from pyoma2.setup import SingleSetup

# ============================================================
# FIXED SETTINGS (no per-record or per-subset tuning)
# ============================================================
FS = 100  # sampling frequency [Hz]
CHANNELS = [f"Sensor{i}" for i in range(1, 11)]
SHORT_NAMES = [c.replace("Sensor", "S") for c in CHANNELS]

OMA_METHOD = "autossi"
CLUSTERING_NAME = "hierarc_avg"

# COV-SSI (base script, opt = 1)
SSI_BR = 120          # Number of block rows in the Hankel matrix
SSI_ORDMIN = 2        # Minimum model order
SSI_ORDMAX = 90       # Maximum model order
SSI_STEP = 2          # Model order increment
SSI_FREQLIM = (2.0, 10.0)

# High-pass filter applied before every OMA run
FILTER_WN = 0.2
FILTER_ORDER = 8
FILTER_BTYPE = "highpass"

# Global clustering (base script, global run)
GLOBAL_LINKAGE = "average"
GLOBAL_DC = "0.07"
GLOBAL_POST_PROC = ["merge_similar", "fn_IQR", "min_size_pctg"]
GLOBAL_MIN_PCTG = 0.27
GLOBAL_SELECT = "avg"

# Subset clustering (base script, subset runs; min_pctg = 0.2 for ALL subsets)
SUBSET_LINKAGE = "complete"
SUBSET_DISTANCE = ["dlambda"]
SUBSET_DC = "0.049"
SUBSET_POST_PROC = ["fn_IQR", "min_size_pctg"]
SUBSET_MIN_PCTG = 0.2
SUBSET_SELECT = "avg"

# Frozen subsets (0-based column indices): {S2,S3,S6}, {S1,S2,S8}, {S2,S3,S8}
SUBSETS = [[1, 2, 5], [0, 1, 7], [1, 2, 7]]

# Mode matching (base script)
MATCH_REMOVE_MATCHED_MODES = True
MATCH_FACTOR_FREQ = 0.98
MATCH_FACTOR_MAC = 0.02
SUBSET_FREQ_PREFILTER_HZ = 20.0  # base script: matching_mask = ~(Fn_subset >= 20)
MAC_THRESHOLD = 0.8              # reliable mode: matched mode with MAC >= MAC_THRESHOLD

# Record exclusion criteria
FLAT_STD_RATIO = 1e-3    # (b) channel std below FLAT_STD_RATIO * median std of the 10 channels
MIN_LENGTH_RATIO = 0.9   # (c) record length below MIN_LENGTH_RATIO * training record length

# Sanity check on the training record (global OMA)
EXPECTED_TRAINING_FREQS_HZ = [2.873, 3.818, 4.505, 5.069, 5.232, 5.407,
                              6.085, 6.457, 6.884, 7.266, 8.276, 9.555]
SANITY_FREQ_TOL_HZ = 1e-3

# Optional per-mode tracking to the training global modes
DO_MODE_TRACKING = True
TRACKING_MAC_THRESHOLD = 0.8

# Ranking: values closer than this are treated as a tie
TIE_ATOL = 1e-9
RANKING_CRITERIA = ["MAC_min", "frac_reliable"]

# Paths (training file: same path logic as the base script)
TRAINING_FILE_CANDIDATES = [
    pathlib.Path(r"C:\Users\melig\Desktop\OMA\Roberto_signal_Puente_Circunv.xlsx"),
    pathlib.Path(r"C:\Users\User\Documents\DOCTORADO\Francesca\MendezNuñez\Roberto_signal_Puente_Circunv.xlsx"),
]
TEST_FILE_PATTERN = "Accelerations_*.xlsx"  # searched in the training file's folder

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "oos_bridge"
FIGURES_DIR = OUTPUT_DIR / "plots"

# Output table layouts
SUBSET_ROW_COLUMNS = [
    "role", "record", "subset_id", "subset_name", "subset_sensors", "status", "error",
    "n_global_modes", "n_subset_modes", "n_subset_modes_prefiltered",
    "n_matched", "n_reliable", "frac_reliable",
    "mean_abs_df_pct", "max_abs_df_pct", "MAC_min", "MAC_mean",
]
MODE_ROW_COLUMNS = [
    "role", "record", "subset_id", "subset_name", "subset_sensors",
    "subset_mode", "f_subset_Hz", "global_mode", "f_global_Hz",
    "df_pct", "abs_df_pct", "MAC", "reliable",
    "training_mode", "f_training_Hz", "tracking_MAC",
]
TRACKING_ROW_COLUMNS = [
    "role", "record", "global_mode", "f_global_Hz", "training_mode", "f_training_Hz", "tracking_MAC",
]
AGGREGATED_METRICS = [
    "n_global_modes", "n_matched", "n_reliable", "frac_reliable",
    "mean_abs_df_pct", "max_abs_df_pct", "MAC_min", "MAC_mean",
]

logger = logging.getLogger("oos_bridge")


# ============================================================
# SET-UP, PATHS AND SETTINGS
# ============================================================
def setup_output_and_logging():
    """Create the output folders and log to both the console and run_log.txt."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(OUTPUT_DIR / "run_log.txt", mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def _record_number(path):
    match = re.search(r"_(\d+)$", path.stem)
    return (int(match.group(1)) if match else np.inf, path.name)


def resolve_paths():
    """Training file (base-script path logic) and the test files in the same folder."""
    training_file = next((p for p in TRAINING_FILE_CANDIDATES if p.exists()), None)
    if training_file is None:
        raise FileNotFoundError(
            f"Training record not found in any of: {[str(p) for p in TRAINING_FILE_CANDIDATES]}")
    test_files = sorted(training_file.parent.glob(TEST_FILE_PATTERN), key=_record_number)
    return training_file, test_files


def sha256_of_file(path, chunk_size=1 << 20):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def subset_meta(subset_id, subset):
    return {
        "subset_id": subset_id,
        "subset_name": "subset_" + "_".join(map(str, subset)),
        "subset_sensors": "-".join(SHORT_NAMES[i] for i in subset),
    }


def write_settings(training_file, training_sha256, n_samples_training, test_files_info):
    settings = {
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "FS": FS,
        "CHANNELS": CHANNELS,
        "OMA_METHOD": OMA_METHOD,
        "SSI": {"method": "cov", "br": SSI_BR, "ordmin": SSI_ORDMIN, "ordmax": SSI_ORDMAX,
                "step": SSI_STEP, "freqlim": list(SSI_FREQLIM), "calc_unc": False},
        "filter": {"Wn": FILTER_WN, "order": FILTER_ORDER, "btype": FILTER_BTYPE},
        "global_clustering": {"step1": {"hc": False, "sc": False, "pre_cluster": False, "pre_clus_typ": "GMM"},
                              "algo": "hierarc", "linkage": GLOBAL_LINKAGE, "dc": GLOBAL_DC,
                              "post_proc": GLOBAL_POST_PROC, "min_pctg": GLOBAL_MIN_PCTG,
                              "freqlim": list(SSI_FREQLIM), "select": GLOBAL_SELECT},
        "subset_clustering": {"step1": {"hc": False, "sc": False, "pre_cluster": False, "pre_clus_typ": "GMM"},
                              "algo": "hierarc", "linkage": SUBSET_LINKAGE, "distance": SUBSET_DISTANCE,
                              "dc": SUBSET_DC, "post_proc": SUBSET_POST_PROC, "min_pctg": SUBSET_MIN_PCTG,
                              "freqlim": list(SSI_FREQLIM), "select": SUBSET_SELECT},
        "SUBSETS": [{**subset_meta(k, s), "indices": s} for k, s in enumerate(SUBSETS, start=1)],
        "matching": {"function": "helpers.outils.match_modes_fem_oma_II",
                     "remove_matched_modes": MATCH_REMOVE_MATCHED_MODES,
                     "factor_freq": MATCH_FACTOR_FREQ, "factor_mac": MATCH_FACTOR_MAC,
                     "subset_freq_prefilter": f"keep subset modes with ~(Fn >= {SUBSET_FREQ_PREFILTER_HZ})",
                     "reference_shapes": "global mode shapes of the same record restricted to the subset rows"},
        "MAC_THRESHOLD": MAC_THRESHOLD,
        "exclusion": {"a": "any NaN or inf in the 10 channels",
                      "b": f"any channel std < {FLAT_STD_RATIO} * median std of the 10 channels",
                      "c": f"record length < {MIN_LENGTH_RATIO} * training record length",
                      "FLAT_STD_RATIO": FLAT_STD_RATIO, "MIN_LENGTH_RATIO": MIN_LENGTH_RATIO},
        "sanity_check": {"EXPECTED_TRAINING_FREQS_HZ": EXPECTED_TRAINING_FREQS_HZ,
                         "SANITY_FREQ_TOL_HZ": SANITY_FREQ_TOL_HZ},
        "mode_tracking": {"DO_MODE_TRACKING": DO_MODE_TRACKING,
                          "TRACKING_MAC_THRESHOLD": TRACKING_MAC_THRESHOLD,
                          "rule": "full 10-channel MAC to the training global modes, "
                                  "one-to-one, highest MAC first"},
        "ranking": {"criteria": RANKING_CRITERIA, "TIE_ATOL": TIE_ATOL,
                    "rule": "per test record, only subsets with status ok and a finite value are compared"},
        "training_file": {"file": training_file.name, "path": str(training_file),
                          "sha256": training_sha256, "n_samples": n_samples_training},
        "test_files": test_files_info,
    }
    with open(OUTPUT_DIR / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ============================================================
# LOADING AND EXCLUSION CHECK
# ============================================================
def load_record(path):
    """
    Read one record and select Sensor1..Sensor10 by name. Rows are NOT dropped,
    so that any NaN/inf reaches the exclusion check.
    """
    df = pd.read_excel(path)
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing channels {missing}")
    df_sel = df[CHANNELS].apply(pd.to_numeric, errors="coerce")
    return df_sel.to_numpy(dtype=float)


def check_exclusion(data, n_samples_training):
    """Return the exclusion reason(s) as a string, or None if the record is accepted."""
    reasons = []

    # (a) NaN or inf in any of the 10 channels
    finite = np.isfinite(data)
    if not finite.all():
        bad = [CHANNELS[j] for j in np.where(~finite.all(axis=0))[0]]
        reasons.append(f"(a) NaN/inf values in {', '.join(bad)} "
                       f"({int((~finite).sum())} non-finite samples)")

    # (b) flat channel: std below FLAT_STD_RATIO times the median std of the 10 channels
    with np.errstate(invalid="ignore"):
        stds = np.nanstd(np.where(finite, data, np.nan), axis=0)
    median_std = np.nanmedian(stds)
    flat = np.where(stds < FLAT_STD_RATIO * median_std)[0]
    if flat.size:
        details = ", ".join(f"{CHANNELS[j]} (std = {stds[j]:.3e})" for j in flat)
        reasons.append(f"(b) flat channel(s): {details}; median std = {median_std:.3e}")

    # (c) record shorter than MIN_LENGTH_RATIO times the training record
    min_length = MIN_LENGTH_RATIO * n_samples_training
    if data.shape[0] < min_length:
        reasons.append(f"(c) record length {data.shape[0]} < {min_length:.0f} samples "
                       f"({MIN_LENGTH_RATIO:.0%} of the training length {n_samples_training})")

    return "; ".join(reasons) if reasons else None


# ============================================================
# OMA (new Step/Clustering objects for every run)
# ============================================================
def make_run_params():
    return SSIRunParams(
        br=SSI_BR,
        method="cov",
        ordmin=SSI_ORDMIN,
        ordmax=SSI_ORDMAX,
        step=SSI_STEP,
        calc_unc=False
    )


def make_step1():
    return Step1(
        hc=False,
        sc=False,
        pre_cluster=False,
        pre_clus_typ="GMM"
    )


def make_global_clustering():
    step2 = Step2(
        algo="hierarc",
        linkage=GLOBAL_LINKAGE,
        dc=GLOBAL_DC
    )
    step3 = Step3(
        post_proc=list(GLOBAL_POST_PROC),
        min_pctg=GLOBAL_MIN_PCTG,
        freqlim=SSI_FREQLIM,
        select=GLOBAL_SELECT
    )
    return Clustering(name=CLUSTERING_NAME, steps=(make_step1(), step2, step3))


def make_subset_clustering():
    step2 = Step2(
        algo="hierarc",
        linkage=SUBSET_LINKAGE,
        distance=list(SUBSET_DISTANCE),
        dc=SUBSET_DC
    )
    step3 = Step3(
        post_proc=list(SUBSET_POST_PROC),
        min_pctg=SUBSET_MIN_PCTG,
        freqlim=SSI_FREQLIM,
        select=SUBSET_SELECT
    )
    return Clustering(name=CLUSTERING_NAME, steps=(make_step1(), step2, step3))


def _run_cov_ssi(data, clustering):
    """High-pass filter + COV-SSI + clustering, as in the base script."""
    setup = SingleSetup(data, fs=FS)
    setup.filter_data(Wn=FILTER_WN, order=FILTER_ORDER, btype=FILTER_BTYPE)

    oma_alg = SSI(name=OMA_METHOD, run_params=make_run_params())
    oma_alg.add_clustering(clustering)

    setup.add_algorithms(oma_alg)
    setup.run_by_name(OMA_METHOD)
    setup[OMA_METHOD].run_all_clustering()

    clus_res = oma_alg.result.clustering_results[CLUSTERING_NAME]
    Fn = np.asarray(clus_res.Fn, dtype=float).ravel()
    Phi = np.asarray(clus_res.Phi)
    if Phi.ndim == 1:  # single mode or no modes: keep the (n_channels, n_modes) layout
        Phi = Phi.reshape(data.shape[1], -1)
    return oma_alg, Fn, Phi


def _save_stab_cluster_plot(oma_alg, filepath, title, **plot_kwargs):
    """Save the stabilisation-cluster plot; a plotting error must not discard the OMA result."""
    try:
        fig, ax = oma_alg.plot_stab_cluster(CLUSTERING_NAME, **plot_kwargs)
        ax.set_title(title)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"Plot not saved ({filepath.name}): {type(exc).__name__}: {exc}")
    finally:
        plt.close("all")


def run_global_oma(data, record):
    oma_alg, Fn, Phi = _run_cov_ssi(data, make_global_clustering())
    _save_stab_cluster_plot(
        oma_alg,
        FIGURES_DIR / f"{record}_global_stab_cluster.png",
        f"{record} | {OMA_METHOD} | Global (10 channels) | Stabilisation clustering",
    )
    return Fn, Phi


def run_subset_oma(data, subset, record, meta):
    oma_alg, Fn, Phi = _run_cov_ssi(data[:, subset], make_subset_clustering())
    sensor_ids = ", ".join(SHORT_NAMES[i] for i in subset)
    _save_stab_cluster_plot(
        oma_alg,
        FIGURES_DIR / f"{record}_{meta['subset_name']}_stab_cluster.png",
        f"{record} | {OMA_METHOD} | Sensors: {sensor_ids} | Stabilisation clustering",
        plot_noise=True,
        freqlim=SSI_FREQLIM,
    )
    return Fn, Phi


# ============================================================
# MATCHING AND METRICS
# ============================================================
def match_subset_to_global(Fn_subset, Phi_subset, Fn_global, Phi_global, subset):
    """
    Match the subset modes to the global modes of the same record, exactly as in
    the base script. Note that match_modes_fem_oma_II assigns EVERY subset mode to
    a global mode, so "matched" means "assigned"; quality is judged through MAC.

    Returns (mode_rows, n_prefiltered, error); error is None on success, or the
    exception text if the matching function raised (matching_failed).
    """
    matching_mask = ~(Fn_subset >= SUBSET_FREQ_PREFILTER_HZ)
    Fn_subset_match = Fn_subset[matching_mask]
    Phi_subset_match = Phi_subset[:, matching_mask]
    Phi_global_subset = Phi_global[subset, :]

    subset_mode_labels = [f"Subset_Mode_{i + 1}" for i in range(len(Fn_subset_match))]

    try:
        matched_info, _ = outils.match_modes_fem_oma_II(
            frequencies_oma=Fn_subset_match,
            modes_oma=Phi_subset_match,
            frequencies_model=Fn_global,
            modes_model=Phi_global_subset,
            oma_labels=subset_mode_labels,
            remove_matched_modes=MATCH_REMOVE_MATCHED_MODES,
            factor_freq=MATCH_FACTOR_FREQ,
            factor_mac=MATCH_FACTOR_MAC,
            label_FEM="GLOBAL",
        )
    except Exception as exc:
        return [], len(Fn_subset_match), f"{type(exc).__name__}: {exc}"

    mode_rows = []
    for i, label in enumerate(subset_mode_labels):
        info = matched_info[label]
        f_subset = float(Fn_subset_match[i])
        f_global = float(info["Frequency_GLOBAL"])
        mac = float(info["MAC"])
        df_pct = 100.0 * (f_subset - f_global) / f_global  # relative to the global frequency
        mode_rows.append({
            "subset_mode": label,
            "f_subset_Hz": f_subset,
            "global_mode": info["Mode_GLOBAL"],
            "f_global_Hz": f_global,
            "df_pct": df_pct,
            "abs_df_pct": abs(df_pct),
            "MAC": mac,
            "reliable": bool(mac >= MAC_THRESHOLD),
        })
    return mode_rows, len(Fn_subset_match), None


def compute_subset_metrics(mode_rows, n_global):
    n_matched = len(mode_rows)
    n_reliable = int(sum(r["reliable"] for r in mode_rows))
    abs_df = np.array([r["abs_df_pct"] for r in mode_rows], dtype=float)
    macs = np.array([r["MAC"] for r in mode_rows], dtype=float)
    return {
        "n_matched": n_matched,
        "n_reliable": n_reliable,
        "frac_reliable": n_reliable / n_global if n_global > 0 else np.nan,
        "mean_abs_df_pct": abs_df.mean() if n_matched else np.nan,
        "max_abs_df_pct": abs_df.max() if n_matched else np.nan,
        "MAC_min": macs.min() if n_matched else np.nan,
        "MAC_mean": macs.mean() if n_matched else np.nan,
    }


# ------------------------------------------------------------
# OPTIONAL: per-mode tracking to the TRAINING global modes
# ------------------------------------------------------------
def track_to_training_modes(Fn_record, Phi_record, Fn_training, Phi_training):
    """
    OPTIONAL (switch: DO_MODE_TRACKING).
    Map each global mode of a record to the global modes of the training record
    using the MAC on the full 10 channels. One-to-one assignment, highest MAC
    first: the pair with the largest remaining MAC is assigned, its row and column
    are removed, and so on while MAC >= TRACKING_MAC_THRESHOLD. Frequencies are
    not used. Returns one row per global mode of the record.
    """
    mac = np.atleast_2d(outils.MaC(Phi_record, Phi_training))  # (n_record, n_training)
    mac_work = np.where(np.isfinite(mac), mac, -np.inf)
    assignment = {}
    while mac_work.size and mac_work.max() >= TRACKING_MAC_THRESHOLD:
        i, j = np.unravel_index(np.argmax(mac_work), mac_work.shape)
        assignment[i] = j
        mac_work[i, :] = -np.inf
        mac_work[:, j] = -np.inf

    rows = []
    for i, f_record in enumerate(Fn_record):
        j = assignment.get(i)
        rows.append({
            "global_mode": f"Mode_{i + 1}",
            "f_global_Hz": float(f_record),
            "training_mode": f"Mode_{j + 1}" if j is not None else None,
            "f_training_Hz": float(Fn_training[j]) if j is not None else np.nan,
            "tracking_MAC": float(mac[i, j]) if j is not None else np.nan,
        })
    return rows


def mode_recovery_counts(df_modes_test, df_tracking_test, n_records_processed, Fn_training):
    """
    OPTIONAL (switch: DO_MODE_TRACKING).
    Per subset and training mode k: in how many processed test records the global
    OMA contains mode k, and in how many the subset recovers it reliably.
    """
    rows = []
    for subset_id, subset in enumerate(SUBSETS, start=1):
        meta = subset_meta(subset_id, subset)
        reliable = df_modes_test[(df_modes_test["subset_name"] == meta["subset_name"])
                                 & df_modes_test["reliable"].astype(bool)]
        for k, f_training in enumerate(Fn_training):
            label = f"Mode_{k + 1}"
            rows.append({
                **meta,
                "training_mode": label,
                "f_training_Hz": float(f_training),
                "n_records_processed": n_records_processed,
                "n_records_in_global": df_tracking_test.loc[
                    df_tracking_test["training_mode"] == label, "record"].nunique(),
                "n_records_recovered_reliable": reliable.loc[
                    reliable["training_mode"] == label, "record"].nunique(),
            })
    return pd.DataFrame(rows)


# ============================================================
# ONE RECORD: GLOBAL OMA + SUBSETS + MATCHING
# ============================================================
def process_record(record, data, role, training_modes=None):
    """
    Run the full pipeline on one (already accepted) record.
    training_modes: (Fn, Phi) of the training global OMA, used for the optional
    tracking. If None (training record itself), the record's own global modes are used.
    """
    out = {"global_ok": False, "global_error": None, "Fn_global": None, "Phi_global": None,
           "subset_rows": [], "mode_rows": [], "tracking_rows": [], "modal": {}}

    logger.info(f"[{record}] global OMA ({data.shape[0]} samples x {data.shape[1]} channels)")
    try:
        Fn_global, Phi_global = run_global_oma(data, record)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error(f"[{record}] global OMA failed: {error}")
        out["global_error"] = error
        for subset_id, subset in enumerate(SUBSETS, start=1):
            out["subset_rows"].append({"role": role, "record": record, **subset_meta(subset_id, subset),
                                       "status": "global_oma_failed", "error": error})
        return out

    n_global = len(Fn_global)
    out.update(global_ok=True, Fn_global=Fn_global, Phi_global=Phi_global)
    out["modal"]["global"] = {"Fn_Hz": Fn_global, "Phi": Phi_global}
    logger.info(f"[{record}] global modes ({n_global}): {np.round(Fn_global, 3).tolist()}")

    # OPTIONAL: tracking of the record's global modes to the training global modes
    tracking_lookup = {}
    if DO_MODE_TRACKING:
        Fn_ref, Phi_ref = training_modes if training_modes is not None else (Fn_global, Phi_global)
        for row in track_to_training_modes(Fn_global, Phi_global, Fn_ref, Phi_ref):
            out["tracking_rows"].append({"role": role, "record": record, **row})
            tracking_lookup[row["global_mode"]] = row

    for subset_id, subset in enumerate(SUBSETS, start=1):
        meta = subset_meta(subset_id, subset)
        row = {"role": role, "record": record, **meta, "n_global_modes": n_global}

        logger.info(f"[{record}] {meta['subset_name']} ({meta['subset_sensors']}) OMA")
        try:
            Fn_subset, Phi_subset = run_subset_oma(data, subset, record, meta)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error(f"[{record}] {meta['subset_name']}: subset OMA failed: {error}")
            row.update(status="oma_failed", error=error)
            out["subset_rows"].append(row)
            continue

        out["modal"][meta["subset_name"]] = {"Fn_Hz": Fn_subset, "Phi": Phi_subset}
        mode_rows, n_prefiltered, error = match_subset_to_global(
            Fn_subset, Phi_subset, Fn_global, Phi_global, subset)
        row.update(n_subset_modes=len(Fn_subset), n_subset_modes_prefiltered=n_prefiltered)

        if error is not None:
            logger.warning(f"[{record}] {meta['subset_name']}: matching_failed "
                           f"({n_prefiltered} subset modes vs {n_global} global modes): {error}")
            row.update(status="matching_failed", error=error)
            out["subset_rows"].append(row)
            continue

        row.update(status="ok", error="", **compute_subset_metrics(mode_rows, n_global))
        out["subset_rows"].append(row)

        for mode_row in mode_rows:
            tracked = tracking_lookup.get(mode_row["global_mode"], {})
            out["mode_rows"].append({
                "role": role, "record": record, **meta, **mode_row,
                "training_mode": tracked.get("training_mode"),
                "f_training_Hz": tracked.get("f_training_Hz", np.nan),
                "tracking_MAC": tracked.get("tracking_MAC", np.nan),
            })

        logger.info(f"[{record}] {meta['subset_name']}: {row['n_matched']} matched, "
                    f"{row['n_reliable']} reliable (MAC >= {MAC_THRESHOLD}), "
                    f"MAC_min = {row['MAC_min']:.3f}, max|df| = {row['max_abs_df_pct']:.2f} %")
    return out


# ============================================================
# SANITY CHECK
# ============================================================
def sanity_check_training(Fn_global):
    expected = np.asarray(EXPECTED_TRAINING_FREQS_HZ, dtype=float)
    n_rows = max(len(expected), len(Fn_global))
    rows = []
    for k in range(n_rows):
        f_expected = expected[k] if k < len(expected) else np.nan
        f_obtained = float(Fn_global[k]) if k < len(Fn_global) else np.nan
        diff = f_obtained - f_expected
        rows.append({
            "mode": f"Mode_{k + 1}",
            "f_expected_Hz": f_expected,
            "f_obtained_Hz": f_obtained,
            "diff_Hz": diff,
            "within_tol": bool(np.abs(diff) <= SANITY_FREQ_TOL_HZ),
        })
    df = pd.DataFrame(rows)
    passed = len(Fn_global) == len(expected) and bool(df["within_tol"].all())
    return df, passed


# ============================================================
# AGGREGATION ACROSS TEST RECORDS
# ============================================================
def rank_subsets(df_test):
    """
    Per test record and criterion, the subset(s) with the highest value. Only
    subsets with status ok and a finite value are compared. A unique best counts
    as a win; a shared best counts as a tie for every subset sharing it.
    """
    counts = {}
    for subset_id, subset in enumerate(SUBSETS, start=1):
        name = subset_meta(subset_id, subset)["subset_name"]
        counts[name] = {}
        for crit in RANKING_CRITERIA:
            counts[name].update({f"rank_{crit}_wins": 0, f"rank_{crit}_ties": 0,
                                 f"rank_{crit}_n_records_compared": 0})

    for _, grp in df_test.groupby("record"):
        for crit in RANKING_CRITERIA:
            valid = grp[(grp["status"] == "ok") & np.isfinite(grp[crit].astype(float))]
            if valid.empty:
                continue
            values = valid[crit].to_numpy(dtype=float)
            best = np.isclose(values, values.max(), rtol=0.0, atol=TIE_ATOL)
            kind = "wins" if best.sum() == 1 else "ties"
            for name, is_best in zip(valid["subset_name"], best):
                counts[name][f"rank_{crit}_n_records_compared"] += 1
                if is_best:
                    counts[name][f"rank_{crit}_{kind}"] += 1

    return pd.DataFrame.from_dict(counts, orient="index").rename_axis("subset_name").reset_index()


def aggregate(df_test):
    """Median, min, max, Q1 and Q3 of every metric per subset, plus ranking counts."""
    rows = []
    for subset_id, subset in enumerate(SUBSETS, start=1):
        meta = subset_meta(subset_id, subset)
        grp = df_test[df_test["subset_name"] == meta["subset_name"]]
        row = {
            **meta,
            "n_records": grp["record"].nunique(),
            "n_ok": int((grp["status"] == "ok").sum()),
            "n_matching_failed": int((grp["status"] == "matching_failed").sum()),
            "n_oma_failed": int(grp["status"].isin(["oma_failed", "global_oma_failed"]).sum()),
        }
        for metric in AGGREGATED_METRICS:
            # Failed cases carry NaN metrics and are therefore left out here
            values = grp[metric].astype(float).dropna().to_numpy()
            if values.size:
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                stats = {"median": median, "min": values.min(), "max": values.max(), "Q1": q1, "Q3": q3}
            else:
                stats = dict.fromkeys(["median", "min", "max", "Q1", "Q3"], np.nan)
            row[f"{metric}_n"] = int(values.size)
            row.update({f"{metric}_{stat}": value for stat, value in stats.items()})
        rows.append(row)

    return pd.DataFrame(rows).merge(rank_subsets(df_test), on="subset_name", how="left")


# ============================================================
# OUTPUT
# ============================================================
def write_tables(subset_rows, mode_rows, tracking_rows):
    df_subsets = pd.DataFrame(subset_rows, columns=SUBSET_ROW_COLUMNS)
    df_modes = pd.DataFrame(mode_rows, columns=MODE_ROW_COLUMNS)
    df_subsets.to_csv(OUTPUT_DIR / "per_record_subset_metrics.csv", index=False)
    df_modes.to_csv(OUTPUT_DIR / "per_mode_matching.csv", index=False)
    if DO_MODE_TRACKING:
        pd.DataFrame(tracking_rows, columns=TRACKING_ROW_COLUMNS).to_csv(
            OUTPUT_DIR / "global_mode_tracking.csv", index=False)
    return df_subsets, df_modes


# ============================================================
# MAIN
# ============================================================
def main():
    setup_output_and_logging()
    training_file, test_files = resolve_paths()
    logger.info(f"Training record: {training_file}")
    logger.info(f"Test records found: {len(test_files)} ({', '.join(p.name for p in test_files)})")

    # ------------------------------------------------------------
    # 1. Training record: load, check, settings
    # ------------------------------------------------------------
    data_training = load_record(training_file)
    n_samples_training = data_training.shape[0]
    training_reason = check_exclusion(data_training, n_samples_training)
    if training_reason is not None:
        raise RuntimeError(f"The training record fails the exclusion check: {training_reason}")

    logger.info("Hashing files (SHA-256)")
    test_files_info = [{"file": p.name, "path": str(p), "sha256": sha256_of_file(p)} for p in test_files]
    write_settings(training_file, sha256_of_file(training_file), n_samples_training, test_files_info)

    # ------------------------------------------------------------
    # 2. Sanity check: full pipeline on the training record
    # ------------------------------------------------------------
    logger.info("=== SANITY CHECK: training record ===")
    training_result = process_record(training_file.stem, data_training, "training")
    if not training_result["global_ok"]:
        raise RuntimeError(f"Global OMA failed on the training record: {training_result['global_error']}")

    df_sanity, sanity_passed = sanity_check_training(training_result["Fn_global"])
    df_sanity.to_csv(OUTPUT_DIR / "sanity_check_training.csv", index=False)
    logger.info("Training global frequencies vs expected:\n" + df_sanity.to_string(index=False))

    subset_rows = list(training_result["subset_rows"])
    mode_rows = list(training_result["mode_rows"])
    tracking_rows = list(training_result["tracking_rows"])
    modal_results = {training_file.stem: training_result["modal"]}
    write_tables(subset_rows, mode_rows, tracking_rows)

    if not sanity_passed:
        logger.error("SANITY CHECK FAILED: the training global OMA does not reproduce the expected "
                     f"frequencies (tolerance {SANITY_FREQ_TOL_HZ} Hz). Test records not processed.")
        raise SystemExit(1)
    logger.info("Sanity check passed.")
    training_modes = (training_result["Fn_global"], training_result["Phi_global"])
    del data_training

    # ------------------------------------------------------------
    # 3. Exclusion check on all test records (before any test OMA)
    # ------------------------------------------------------------
    logger.info("=== EXCLUSION CHECK: test records ===")
    accepted, excluded = [], []
    for path in test_files:
        data = load_record(path)
        reason = check_exclusion(data, n_samples_training)
        if reason is None:
            logger.info(f"{path.name}: accepted ({data.shape[0]} samples)")
            accepted.append((path, data))
        else:
            logger.warning(f"{path.name}: EXCLUDED - {reason}")
            excluded.append({"file": path.name, "reason": reason})
    pd.DataFrame(excluded, columns=["file", "reason"]).to_csv(OUTPUT_DIR / "excluded_records.csv", index=False)

    # ------------------------------------------------------------
    # 4. Test records
    # ------------------------------------------------------------
    logger.info("=== OUT-OF-SAMPLE RUNS ===")
    global_failures = []
    while accepted:
        path, data = accepted.pop(0)
        result = process_record(path.stem, data, "test", training_modes)
        if not result["global_ok"]:
            global_failures.append({"record": path.stem, "error": result["global_error"]})
        subset_rows.extend(result["subset_rows"])
        mode_rows.extend(result["mode_rows"])
        tracking_rows.extend(result["tracking_rows"])
        modal_results[path.stem] = result["modal"]
        write_tables(subset_rows, mode_rows, tracking_rows)  # partial results after every record

    df_subsets, df_modes = write_tables(subset_rows, mode_rows, tracking_rows)
    outils.save_json_serialized(modal_results, OUTPUT_DIR / "modal_results.json")

    # ------------------------------------------------------------
    # 5. Aggregation across test records
    # ------------------------------------------------------------
    df_test = df_subsets[df_subsets["role"] == "test"]
    processed_records = sorted(set(df_test["record"]) - {f["record"] for f in global_failures},
                               key=lambda r: _record_number(pathlib.Path(r)))
    df_aggregated = aggregate(df_test)
    df_aggregated.to_csv(OUTPUT_DIR / "aggregated_metrics.csv", index=False)

    if DO_MODE_TRACKING:
        df_tracking = pd.DataFrame(tracking_rows, columns=TRACKING_ROW_COLUMNS)
        df_recovery = mode_recovery_counts(
            df_modes[df_modes["role"] == "test"],
            df_tracking[df_tracking["role"] == "test"],
            len(processed_records),
            training_modes[0],
        )
        df_recovery.to_csv(OUTPUT_DIR / "mode_recovery_counts.csv", index=False)

    # ------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------
    subset_oma_failures = df_test[df_test["status"] == "oma_failed"]
    matching_failures = df_test[df_test["status"] == "matching_failed"]

    lines = ["", "=" * 60, "SUMMARY", "=" * 60,
             f"Test records found:     {len(test_files)}",
             f"Excluded:               {len(excluded)}"]
    lines += [f"    - {e['file']}: {e['reason']}" for e in excluded]
    lines += [f"Processed:              {len(processed_records)} ({', '.join(processed_records)})",
              f"Global OMA failures:    {len(global_failures)}"]
    lines += [f"    - {f['record']}: {f['error']}" for f in global_failures]
    lines += [f"Subset OMA failures:    {len(subset_oma_failures)}"]
    lines += [f"    - {r.record} | {r.subset_name} ({r.subset_sensors}): {r.error}"
              for r in subset_oma_failures.itertuples()]
    lines += [f"Matching failures:      {len(matching_failures)}"]
    lines += [f"    - {r.record} | {r.subset_name} ({r.subset_sensors}): "
              f"{r.n_subset_modes_prefiltered:.0f} subset vs {r.n_global_modes:.0f} global modes | {r.error}"
              for r in matching_failures.itertuples()]
    lines += ["", "Per-subset medians over the test records [Q1, Q3]:"]
    for r in df_aggregated.itertuples():
        lines.append(
            f"    {r.subset_name} ({r.subset_sensors}): "
            f"MAC_min = {r.MAC_min_median:.3f} [{r.MAC_min_Q1:.3f}, {r.MAC_min_Q3:.3f}], "
            f"frac_reliable = {r.frac_reliable_median:.3f} [{r.frac_reliable_Q1:.3f}, {r.frac_reliable_Q3:.3f}], "
            f"best MAC_min: {r.rank_MAC_min_wins} wins + {r.rank_MAC_min_ties} ties, "
            f"best frac_reliable: {r.rank_frac_reliable_wins} wins + {r.rank_frac_reliable_ties} ties"
        )
    lines.append(f"\nOutputs written to: {OUTPUT_DIR}")
    logger.info("\n".join(lines))


if __name__ == "__main__":
    main()
