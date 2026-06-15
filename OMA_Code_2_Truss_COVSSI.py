
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
import json
import helpers.outils as outils

from pyoma2.algorithms.ssi import SSI
from pyoma2.algorithms.data.run_params import SSIRunParams, Clustering, Step1, Step2, Step3

from pyoma2.setup import SingleSetup




# ============================================================
# PARAMETERS
# ============================================================
output_dir = pathlib.Path("./results")
figures_dir = pathlib.Path("./plots")
output_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

OMA_METHOD = "autossi"

# Parameters for automatic COV-SSI
SSI_BR = 70
SSI_ORDMIN = 2
SSI_ORDMAX = 50
SSI_STEP = 2
SSI_FREQLIM = (30.0, 400.0)


# ============================================================
# LOAD DATA
# ============================================================
print("\n=== 1. LOADING DATA FROM CSV ===")

file_path = r"C:\Users\User\Documents\DOCTORADO\Francesca_OMA\data\simulated_records_1000Hz.csv"
fs = 1000  # sampling frequency
df = pd.read_csv(file_path)
Phi_id = df.columns.tolist()

df = df.select_dtypes(include=[np.number])
df = df.dropna()
data = df.values


# ============================================================
# PERFORM OMA
# ============================================================
print(f"\n=== 2. COV-SSI computation using {OMA_METHOD} ===")

Pali_ss = SingleSetup(data, fs=fs)
Pali_ss.filter_data(Wn=0.2, order=8, btype="highpass")

run_param = SSIRunParams(
    br=SSI_BR,
    method="cov",
    ordmin=SSI_ORDMIN,
    ordmax=SSI_ORDMAX,
    step=SSI_STEP,
    calc_unc=False
)

oma_alg = SSI(name=OMA_METHOD, run_params=run_param)

step1 = Step1(
    hc=True,
    hc_dict={
        "xi_max": 0.1,
        "mpc_lim": 0.8,
        "mpd_lim": 0.4,
        "CoV_max": 0.20,
    },
    sc=True,
    sc_dict={
        "err_fn": 0.05,
        "err_xi": 0.1,
        "err_phi": 0.05,
    },
    pre_cluster=False,
    # pre_clus_typ="GMM",
    # pre_clus_dist=["dlambda", "dMAC"],
    # transform=None,
)

step2 = Step2(
    distance=["dlambda", "dMAC"],
    weights=None,
    sqrtsqr=False,
    algo="hierarc",
    dc="auto",
    linkage="average",
    min_size=10,
    n_clusters=None,
)

step3 = Step3(
    post_proc=[
        "merge_similar",
        "damp_IQR",
        "fn_IQR",
        "1xorder",
        "min_size_pctg",
        "MTT",
    ],
    merge_dist="auto",
    min_pctg=0.35,
    select="medoid",
    freqlim=SSI_FREQLIM,
)

clus = Clustering(
    name="hierarc_avg",
    steps=(step1, step2, step3)
)

oma_alg.add_clustering(clus)

Pali_ss.add_algorithms(oma_alg)
Pali_ss.run_by_name(OMA_METHOD)
Pali_ss[OMA_METHOD].run_all_clustering()

fig, ax = oma_alg.plot_stab_cluster(
    "hierarc_avg",
    plot_noise=True,
    freqlim=SSI_FREQLIM
)

ax.set_title(f"{OMA_METHOD} | COV-SSI automatic clustering")
fig.savefig(
    figures_dir / f"{OMA_METHOD}_TRUSS_stab_cluster.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

clus_res = oma_alg.result.clustering_results["hierarc_avg"]

Fn_oma = np.asarray(clus_res.Fn, dtype=float)
Phi_oma = np.asarray(clus_res.Phi)

print(f"\nIdentified frequencies using {OMA_METHOD} [Hz]:")
print(Fn_oma)

# ============================================================
# DERIVED METRICS
# ============================================================
use_real_modes = True

if use_real_modes:
    Phi_FIM_analysis = outils.complex_to_normal_mode(Phi_oma)
else:
    Phi_FIM_analysis = Phi_oma

FIM = Phi_FIM_analysis.conj().T @ Phi_FIM_analysis
detFIM = abs(np.linalg.det(FIM))

Lambda, Psi = np.linalg.eig(FIM)
G = np.multiply(np.dot(Phi_FIM_analysis, Psi), np.dot(Phi_FIM_analysis, Psi))
Fe = np.dot(G, np.linalg.inv(np.diag(Lambda)))
Ed = np.sum(Fe, axis=1)

# ============================================================
# EFI METHOD
# ============================================================
EfI_results = outils.EFI(Phi_FIM_analysis, Phi_id, n_sensors=7)
sensors_selected = EfI_results["results"]["Phi_id"]
Phi_EFI = EfI_results["results"]["Phi"]
fig, ax = outils.plot_MAC_1(outils.MaC_nan(Phi_EFI, Phi_EFI))
print(f"Sensors selected by EFI: {sensors_selected}")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "Fn_oma_Hz": Fn_oma,
    "Phi_oma": Phi_oma,
    "Phi_oma_real": outils.complex_to_normal_mode(Phi_oma),
    "FIM": FIM,
    "detFIM": detFIM,
    "Ed": Ed
}

outils.save_json_serialized(results, output_dir / f"{OMA_METHOD}_TRUSS_results.json")
