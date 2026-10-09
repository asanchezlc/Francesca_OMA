"""
Step-by-step inspection of the global OMA (Mendez-Nunez bridge) with the VS Code debugger.
Read-only: nothing is written to results/.

Every figure is shown with plt.show(block=True): execution waits until you close
the window. Put breakpoints on the code lines right after the "# BP-x" comments,
so that no figure window is open while the debugger is paused.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helpers.outils as outils
import OMA_Code_1_OOS_1_run_oma as v1  # importing it forces the "Agg" backend
from pyoma2.algorithms.ssi import SSI
from pyoma2.algorithms.data.run_params import SSIRunParams, Clustering, Step1, Step2, Step3
from pyoma2.setup import SingleSetup

plt.switch_backend("TkAgg")  # GUI backend for the debugger; TkAgg ships with Python on Windows

# ============================================================
# SESSION SETTINGS (global OMA only; v1 values in comments)
# ============================================================
RECORDS = ["Accelerations_21"]  # record stems to inspect; the training stem is also valid

# COV-SSI
BR = 120        # block rows (v1: 120)
ORDMIN = 2      # minimum model order (v1: 2)
ORDMAX = 110     # maximum model order (v1: 90)
STEP = 2        # model order step (v1: 2)

# Step 1: hard and soft criteria (v1 bridge: both off; dictionaries = truss values)
HC = False
HC_DICT = {"xi_max": 0.1, "mpc_lim": 0.8, "mpd_lim": 0.4, "CoV_max": 0.20}
SC = False
SC_DICT = {"err_fn": 0.05, "err_xi": 0.1, "err_phi": 0.05}

# Step 2: hierarchical clustering
DC = "0.07"        # cut-off distance (v1: "0.07", passed as a string as in v1)
LINKAGE = "average"  # (v1: "average")
DISTANCE = None    # None = pyOMA2 default (v1); e.g. ["dlambda", "dMAC"]
MIN_SIZE = None    # None = pyOMA2 default (v1); absolute minimum cluster size, e.g. 10

# Step 3: post-processing
MIN_PCTG = 0.27    # (v1: 0.27)

SHOW_RAW_STAB = True     # raw stabilisation diagram (all poles)
SHOW_DTOT = True         # distance distribution of the clustering
INSPECT_SUBSETS = False  # also show the subset clusters (frozen v1 settings, inspection only)


# ============================================================
# HELPERS
# ============================================================
def settings_tag():
    tag = f"br={BR}, ord={ORDMIN}-{ORDMAX}/{STEP}, dc={DC}, min_pctg={MIN_PCTG}"
    tag += f", HC={'on' if HC else 'off'}, SC={'on' if SC else 'off'}"
    if DISTANCE is not None:
        tag += f", dist={'+'.join(DISTANCE)}"
    if MIN_SIZE is not None:
        tag += f", min_size={MIN_SIZE}"
    return tag


def global_clustering():
    """v1 global clustering with every main parameter exposed."""
    step1_kwargs = {"hc": HC, "sc": SC, "pre_cluster": False, "pre_clus_typ": "GMM"}
    if HC:
        step1_kwargs["hc_dict"] = dict(HC_DICT)
    if SC:
        step1_kwargs["sc_dict"] = dict(SC_DICT)

    step2_kwargs = {"algo": "hierarc", "linkage": LINKAGE, "dc": DC}
    if DISTANCE is not None:
        step2_kwargs["distance"] = list(DISTANCE)
    if MIN_SIZE is not None:
        step2_kwargs["min_size"] = MIN_SIZE

    step3 = Step3(post_proc=list(v1.GLOBAL_POST_PROC), min_pctg=MIN_PCTG,
                  freqlim=v1.SSI_FREQLIM, select=v1.GLOBAL_SELECT)
    return Clustering(name=v1.CLUSTERING_NAME,
                      steps=(Step1(**step1_kwargs), Step2(**step2_kwargs), step3))


def run_global(data):
    """Global COV-SSI as in v1._run_cov_ssi, with the SSI parameters exposed."""
    setup = SingleSetup(data, fs=v1.FS)
    setup.filter_data(Wn=v1.FILTER_WN, order=v1.FILTER_ORDER, btype=v1.FILTER_BTYPE)

    run_params = SSIRunParams(br=BR, method="cov", ordmin=ORDMIN, ordmax=ORDMAX,
                              step=STEP, calc_unc=False)
    oma_alg = SSI(name=v1.OMA_METHOD, run_params=run_params)
    oma_alg.add_clustering(global_clustering())

    setup.add_algorithms(oma_alg)
    setup.run_by_name(v1.OMA_METHOD)
    setup[v1.OMA_METHOD].run_all_clustering()

    clus_res = oma_alg.result.clustering_results[v1.CLUSTERING_NAME]
    Fn = np.asarray(clus_res.Fn, dtype=float).ravel()
    Phi = np.asarray(clus_res.Phi)
    if Phi.ndim == 1:
        Phi = Phi.reshape(data.shape[1], -1)
    return oma_alg, Fn, Phi


def show_and_wait(method, title, *args, **kwargs):
    """Draw one pyOMA2 figure and wait until its window is closed."""
    try:
        fig, ax = method(*args, **kwargs)
        ax.set_title(title)
        plt.show(block=True)  # execution waits here until you close the window
    except Exception as exc:
        print(f"[plot skipped] {title}: {type(exc).__name__}: {exc}")
    finally:
        plt.close("all")


# ============================================================
# TRAINING MODES (for the 10-channel tracking)
# ============================================================
training_file, test_files = v1.resolve_paths()
all_files = {p.stem: p for p in [training_file, *test_files]}
modal_v1 = outils.load_json_serialized(v1.OUTPUT_DIR / "modal_results.json")
Fn_tr = np.ravel(np.asarray(modal_v1[training_file.stem]["global"]["Fn_Hz"], dtype=float))
Phi_tr = np.asarray(modal_v1[training_file.stem]["global"]["Phi"])
print("Training global modes [Hz]:", np.round(Fn_tr, 3).tolist())
print("Available records:", list(all_files))

# ============================================================
# INSPECTION LOOP
# ============================================================
for record in RECORDS:
    data = v1.load_record(all_files[record])
    print(f"\n=== {record}: {data.shape[0]} samples x {data.shape[1]} channels ===")

    # --- Global OMA (10 channels) -------------------------------------------
    oma_alg, Fn, Phi = run_global(data)
    tag = f"{record} | global | {settings_tag()}"
    print(f"{tag}: {len(Fn)} modes -> {np.round(Fn, 3).tolist()}")  # BP-1: inspect Fn, Phi

    if SHOW_RAW_STAB:
        show_and_wait(oma_alg.plot_stab, f"{tag} | raw stabilisation",
                      freqlim=v1.SSI_FREQLIM, hide_poles=False)
    # BP-2: raw diagram seen and closed

    show_and_wait(oma_alg.plot_stab_cluster, f"{tag} | clusters (grey = discarded poles)",
                  v1.CLUSTERING_NAME, plot_noise=True, freqlim=v1.SSI_FREQLIM)
    # BP-3: cluster diagram seen and closed

    if SHOW_DTOT:
        show_and_wait(oma_alg.plot_dtot_distrib, f"{tag} | distance distribution",
                      v1.CLUSTERING_NAME)

    # --- Which training modes were found (10-channel MAC) --------------------
    track = pd.DataFrame(v1.track_to_training_modes(Fn, Phi, Fn_tr, Phi_tr))
    found = set(track["training_mode"].dropna())
    missing = [f"Mode_{k + 1} ({Fn_tr[k]:.3f} Hz)" for k in range(len(Fn_tr))
               if f"Mode_{k + 1}" not in found]
    n_untracked = int(track["training_mode"].isna().sum())
    print(track.round(3).to_string(index=False))
    print(f"Missing training modes: {missing}")
    print(f"Identified modes not matching any training mode: {n_untracked}")  # BP-4: inspect track

    # --- Optional: subsets (frozen v1 settings, inspection only) ------------
    if INSPECT_SUBSETS:
        for subset in v1.SUBSETS:
            sensors = ", ".join(v1.SHORT_NAMES[i] for i in subset)
            oma_sub, Fn_sub, Phi_sub = v1._run_cov_ssi(data[:, subset], v1.make_subset_clustering())
            print(f"{record} | {sensors}: {len(Fn_sub)} modes -> {np.round(np.sort(Fn_sub), 3).tolist()}")
            show_and_wait(oma_sub.plot_stab_cluster, f"{record} | {sensors} | clusters",
                          v1.CLUSTERING_NAME, plot_noise=True, freqlim=v1.SSI_FREQLIM)
            # BP-5: subset diagram seen and closed

print("\nDone.")