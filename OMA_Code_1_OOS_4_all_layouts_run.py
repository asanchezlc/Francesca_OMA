"""
Subset OMA of ALL 3-sensor layouts (10 choose 3 = 120) in every record - Mendez-Nunez bridge.

Heavy and resumable. For every record stored in results/oos_bridge/modal_results.json
(training + test records) and every 3-sensor layout, the subset COV-SSI is run with
exactly the frozen subset settings of OMA_Code_1_OOS_1_run_oma.py: its functions are
called (v1._run_cov_ssi, v1.make_subset_clustering), no setting is copied. The global
modes are not recomputed (step 5 reads them from modal_results.json).

Store: results/oos_bridge/all_layouts/modes/<record>.jsonl, one JSON line per layout
(frequencies, mode shapes, status), appended as soon as the layout is done. Layouts
already stored, failures included, are skipped on restart.

Sanity check (run first): the 3 Table-6 layouts of every record must reproduce the
subset modes stored in modal_results.json (same number of modes, frequencies within
SANITY_FREQ_TOL_HZ); otherwise the script stops.

Outputs: results/oos_bridge/all_layouts/
"""
import os
os.environ.setdefault("TQDM_DISABLE", "1")  # silence the pyOMA2 progress bars (also in the workers)

import itertools
import json
import logging
import pathlib
import platform
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import threadpoolctl
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import helpers.outils as outils
import OMA_Code_1_OOS_1_run_oma as v1  # frozen OMA settings and shared helpers

# ============================================================
# SETTINGS
# ============================================================
N_JOBS = 8           # parallel worker processes (results do not depend on it, see BLAS_THREADS)
BLAS_THREADS = 1     # BLAS threads per OMA run, fixed for every N_JOBS so that results are identical
SAVE_PLOTS = False   # stabilisation-cluster plot of every (record, layout)

LAYOUT_SIZE = 3
LAYOUTS = [list(c) for c in itertools.combinations(range(len(v1.CHANNELS)), LAYOUT_SIZE)]  # 120
TABLE6_LAYOUTS = [list(s) for s in v1.SUBSETS]  # {S2,S3,S6}, {S1,S2,S8}, {S2,S3,S8}
SANITY_FREQ_TOL_HZ = 1e-6

MODAL_RESULTS_FILE = v1.OUTPUT_DIR / "modal_results.json"
V1_SETTINGS_FILE = v1.OUTPUT_DIR / "settings.json"
OUTPUT_DIR = v1.OUTPUT_DIR / "all_layouts"
MODES_DIR = OUTPUT_DIR / "modes"
FIGURES_DIR = OUTPUT_DIR / "plots"

logger = logging.getLogger("oos_bridge_all_layouts_run")


# ============================================================
# HELPERS
# ============================================================
def setup_output_and_logging():
    MODES_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PLOTS:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(),
                    logging.FileHandler(OUTPUT_DIR / "run_log.txt", mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def layout_id(layout):
    return LAYOUTS.index(list(layout)) + 1


def layout_meta(layout):
    """Same naming as OMA_Code_1_OOS_1_run_oma.py (subset_1_2_5, S2-S3-S6)."""
    return v1.subset_meta(layout_id(layout), list(layout))


def store_file(record):
    return MODES_DIR / f"{record}.jsonl"


def read_store(record):
    """Stored entries of one record, keyed by layout name (the last valid line wins)."""
    path = store_file(record)
    entries = {}
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entry = outils.from_serializable(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"{path.name}: line {n} is incomplete and is ignored (its layout is recomputed)")
                continue
            entries[entry["layout_name"]] = entry
    return entries


def append_to_store(record, entry):
    path = store_file(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(outils.to_serializable(entry)) + "\n")
        f.flush()
        os.fsync(f.fileno())


def repair_store_tail(record):
    """Make sure a new line never continues an incomplete last line left by an interrupted run."""
    path = store_file(record)
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")


def run_layout(data, layout, record):
    """Subset OMA of one layout with the frozen v1 settings. Never raises: failures are returned."""
    meta = layout_meta(layout)
    t0 = time.perf_counter()
    try:
        with threadpool_limits(limits=BLAS_THREADS):
            oma_alg, Fn, Phi = v1._run_cov_ssi(data[:, layout], v1.make_subset_clustering())
        if SAVE_PLOTS:
            sensor_ids = ", ".join(v1.SHORT_NAMES[i] for i in layout)
            v1._save_stab_cluster_plot(
                oma_alg, FIGURES_DIR / f"{record}_{meta['subset_name']}_stab_cluster.png",
                f"{record} | {v1.OMA_METHOD} | Sensors: {sensor_ids} | Stabilisation clustering",
                plot_noise=True, freqlim=v1.SSI_FREQLIM)
        result = {"status": "ok", "error": "", "Fn_Hz": Fn, "Phi": Phi}
    except Exception as exc:
        result = {"status": "oma_failed", "error": f"{type(exc).__name__}: {exc}", "Fn_Hz": [], "Phi": []}
    return {"record": record, "layout": [int(i) for i in layout], "layout_id": meta["subset_id"],
            "layout_name": meta["subset_name"], "layout_sensors": meta["subset_sensors"],
            **result, "elapsed_s": time.perf_counter() - t0}


def run_layouts(jobs, data_by_record):
    """Run (record, layout) jobs, appending each result to its store as soon as it is done."""
    if not jobs:
        return []
    tasks = (delayed(run_layout)(data_by_record[record], layout, record) for record, layout in jobs)
    done = []
    for entry in Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(tasks):
        append_to_store(entry["record"], entry)
        done.append(entry)
        if entry["status"] != "ok":
            logger.warning(f"[{entry['record']}] {entry['layout_name']} ({entry['layout_sensors']}): "
                           f"subset OMA failed: {entry['error']}")
    return done


def check_table6_layouts(records, modal_results):
    """Stored Table-6 layouts vs the subset modes of modal_results.json."""
    rows = []
    for record in records:
        stored = read_store(record)
        for layout in TABLE6_LAYOUTS:
            name = layout_meta(layout)["subset_name"]
            entry = stored.get(name)
            Fn_ref = np.ravel(np.asarray(modal_results[record][name]["Fn_Hz"], dtype=float))
            Fn_new = (np.ravel(np.asarray(entry["Fn_Hz"], dtype=float))
                      if entry is not None and entry["status"] == "ok" else np.array([]))
            same_n = Fn_new.size == Fn_ref.size
            max_df = float(np.max(np.abs(Fn_new - Fn_ref))) if same_n and Fn_ref.size else np.nan
            if same_n and Fn_ref.size:
                Phi_ref = np.asarray(modal_results[record][name]["Phi"])
                max_dphi = float(np.max(np.abs(np.asarray(entry["Phi"]) - Phi_ref)))
            else:
                max_dphi = np.nan
            rows.append({"record": record, "layout_name": name, "n_modes_stored": Fn_new.size,
                         "n_modes_v1": Fn_ref.size, "max_abs_df_Hz": max_df, "max_abs_dPhi": max_dphi,
                         "passed": bool(same_n and (Fn_ref.size == 0 or max_df <= SANITY_FREQ_TOL_HZ))})
    return pd.DataFrame(rows)


def write_settings(records, record_files, v1_settings):
    file_hashes = {r: {"path": str(record_files[r]), "sha256": v1.sha256_of_file(record_files[r])}
                   for r in records}
    settings = {
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "threadpoolctl_version": threadpoolctl.__version__,
        "N_JOBS": N_JOBS,
        "BLAS_THREADS": BLAS_THREADS,
        "SAVE_PLOTS": SAVE_PLOTS,
        "LAYOUT_SIZE": LAYOUT_SIZE,
        "n_layouts": len(LAYOUTS),
        "layout_order": "itertools.combinations(range(10), 3); layout_id = position + 1",
        "TABLE6_LAYOUTS": TABLE6_LAYOUTS,
        "SANITY_FREQ_TOL_HZ": SANITY_FREQ_TOL_HZ,
        "subset_oma": "v1._run_cov_ssi(data[:, layout], v1.make_subset_clustering()) "
                      "(OMA_Code_1_OOS_1_run_oma.py, frozen subset settings)",
        "records": records,
        "inputs": {
            "modal_results": {"path": str(MODAL_RESULTS_FILE), "sha256": v1.sha256_of_file(MODAL_RESULTS_FILE)},
            "v1_settings": {"path": str(V1_SETTINGS_FILE), "sha256": v1.sha256_of_file(V1_SETTINGS_FILE)},
            "records": file_hashes,
        },
        "oma_settings_v1": v1_settings,
    }
    with open(OUTPUT_DIR / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    return file_hashes


# ============================================================
# MAIN
# ============================================================
def main():
    setup_output_and_logging()
    t_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info(f"Run started: {len(LAYOUTS)} layouts, N_JOBS = {N_JOBS}, BLAS_THREADS = {BLAS_THREADS}, "
                f"SAVE_PLOTS = {SAVE_PLOTS}")

    with open(V1_SETTINGS_FILE, encoding="utf-8") as f:
        v1_settings = json.load(f)
    modal_results = outils.load_json_serialized(MODAL_RESULTS_FILE)
    training_record = pathlib.Path(v1_settings["training_file"]["file"]).stem
    records = [training_record] + sorted((r for r in modal_results if r != training_record),
                                         key=lambda r: v1._record_number(pathlib.Path(r)))

    # Record files: same files (SHA-256) as the v1 OMA run
    training_file, test_files = v1.resolve_paths()
    record_files = {p.stem: p for p in [training_file, *test_files]}
    missing = [r for r in records if r not in record_files]
    if missing:
        raise SystemExit(f"Record files not found: {missing}")
    file_hashes = write_settings(records, record_files, v1_settings)
    v1_hashes = {pathlib.Path(t["file"]).stem: t["sha256"] for t in v1_settings["test_files"]}
    v1_hashes[training_record] = v1_settings["training_file"]["sha256"]
    changed = [r for r in records if file_hashes[r]["sha256"] != v1_hashes.get(r)]
    if changed:
        raise SystemExit(f"Record files differ from the v1 OMA run (SHA-256): {changed}")
    logger.info(f"Records ({len(records)}): files identical to the v1 run (SHA-256)")

    for record in records:
        repair_store_tail(record)
    stored = {record: read_store(record) for record in records}
    n_stored = sum(len(s) for s in stored.values())
    logger.info(f"Already stored: {n_stored} of {len(records) * len(LAYOUTS)} (record, layout) pairs")

    # Load each record once (only those with layouts still to run)
    to_load = [r for r in records if len(stored[r]) < len(LAYOUTS)]
    t0 = time.perf_counter()
    loaded = Parallel(n_jobs=min(N_JOBS, max(len(to_load), 1)))(
        delayed(v1.load_record)(record_files[r]) for r in to_load)
    data_by_record = dict(zip(to_load, loaded))
    if to_load:
        logger.info(f"Loaded {len(to_load)} records in {time.perf_counter() - t0:.1f} s")

    # ------------------------------------------------------------
    # 1. Sanity check: the Table-6 layouts of every record (run first)
    # ------------------------------------------------------------
    jobs = [(r, layout) for r in records for layout in TABLE6_LAYOUTS
            if layout_meta(layout)["subset_name"] not in stored[r]]
    t0 = time.perf_counter()
    done = run_layouts(jobs, data_by_record)
    if done:
        wall = time.perf_counter() - t0
        logger.info(f"Table-6 layouts: {len(done)} runs in {wall:.1f} s wall time -> {wall / len(done):.2f} s per run "
                    f"(mean time inside a run: {np.mean([e['elapsed_s'] for e in done]):.2f} s)")

    df_check = check_table6_layouts(records, modal_results)
    df_check.to_csv(OUTPUT_DIR / "sanity_check_table6_layouts.csv", index=False)
    if not df_check["passed"].all():
        logger.error("SANITY CHECK FAILED: stored Table-6 layouts differ from modal_results.json:\n"
                     + df_check[~df_check["passed"]].to_string(index=False))
        raise SystemExit(1)
    logger.info(f"SANITY CHECK PASSED: {len(df_check)} Table-6 (record, layout) pairs reproduce modal_results.json "
                f"(max |df| = {df_check['max_abs_df_Hz'].max():.2e} Hz, max |dPhi| = {df_check['max_abs_dPhi'].max():.2e})")

    # ------------------------------------------------------------
    # 2. All remaining layouts, record by record
    # ------------------------------------------------------------
    for record in records:
        stored_names = set(read_store(record))
        jobs = [(record, layout) for layout in LAYOUTS if layout_meta(layout)["subset_name"] not in stored_names]
        if not jobs:
            logger.info(f"[{record}] all {len(LAYOUTS)} layouts already stored")
            continue
        t0 = time.perf_counter()
        done = run_layouts(jobs, data_by_record)
        wall = time.perf_counter() - t0
        n_failed = sum(e["status"] != "ok" for e in done)
        logger.info(f"[{record}] {len(done)} layouts in {wall:.1f} s ({wall / len(done):.2f} s per run), "
                    f"{n_failed} failed")
        data_by_record.pop(record, None)

    # ------------------------------------------------------------
    # 3. Summary and failures
    # ------------------------------------------------------------
    rows, failures = [], []
    for record in records:
        entries = read_store(record)
        n_ok = sum(e["status"] == "ok" for e in entries.values())
        rows.append({"record": record, "n_layouts_stored": len(entries), "n_ok": n_ok,
                     "n_oma_failed": len(entries) - n_ok,
                     "median_n_modes": float(np.median([np.size(e["Fn_Hz"]) for e in entries.values()
                                                        if e["status"] == "ok"] or [np.nan]))})
        failures += [{"record": record, "layout_name": e["layout_name"], "layout_sensors": e["layout_sensors"],
                      "error": e["error"]} for e in entries.values() if e["status"] != "ok"]
    df_summary = pd.DataFrame(rows)
    pd.DataFrame(failures, columns=["record", "layout_name", "layout_sensors", "error"]).to_csv(
        OUTPUT_DIR / "failures.csv", index=False)
    complete = bool((df_summary["n_layouts_stored"] == len(LAYOUTS)).all())
    logger.info("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60 + "\n" + df_summary.to_string(index=False)
                + f"\nTotal: {int(df_summary['n_layouts_stored'].sum())} of {len(records) * len(LAYOUTS)} pairs stored, "
                f"{int(df_summary['n_oma_failed'].sum())} subset OMA failures; complete: {complete}"
                + f"\nElapsed: {(time.perf_counter() - t_start) / 60:.1f} min\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
