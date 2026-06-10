
import numpy as np
import pandas as pd

import pathlib

import helpers.outils as outils

from pyoma2.algorithms.ssi import SSI
from pyoma2.algorithms.data.run_params import SSIRunParams, Clustering, Step1, Step2, Step3

from pyoma2.setup import SingleSetup

import matplotlib.pyplot as plt
# ============================================================
# PARAMETERS
# ============================================================
output_dir = pathlib.Path("./results")
figures_dir = output_dir / "plots"
output_dir.mkdir(exist_ok=True)

OMA_METHOD = "autossi"

SSI_BR = 90          # Number of block rows in the Hankel matrix
SSI_ORDMIN = 2       # Minimum model order
SSI_ORDMAX = 120      # Maximum model order
SSI_STEP = 2         # Model order increment
SSI_FREQLIM = (2.0, 10.0)


# ============================================================
# LOAD DATA
# ============================================================
print("\n=== 1. LOADING DATA FROM EXCEL ===")

file_path = r"C:\Users\melig\Desktop\OMA\Roberto_signal_Puente_Circunv.xlsx"
fs = 100  # sampling frequency


try:
    df = pd.read_excel(file_path)
except:
    file_path = r"C:\Users\User\Documents\DOCTORADO_CODES\Francesca_OMA\Roberto_signal_Puente_Circunv.xlsx"
    df = pd.read_excel(file_path)


df = df.select_dtypes(include=[np.number])
df = df.dropna()

selected_cols = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5", "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10"]
df_sel = df[selected_cols].copy()

print("\nSelected sensors:")
print(df_sel.columns.tolist())

data = df_sel.values

print("Selected data shape:", data.shape)
print("Number of samples:", data.shape[0])
print("Number of selected channels:", data.shape[1])

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
    hc=False,
    sc=False,
    pre_cluster=False,
    pre_clus_typ="GMM"
)

step2 = Step2(
    algo="hierarc",
    linkage="average",
    dc="auto"
)

step3 = Step3(
    post_proc=["merge_similar", "fn_IQR", "min_size_pctg"],
    min_pctg=0.15,
    freqlim=SSI_FREQLIM,
    select="medoid"
)

clus = Clustering(
    name="hierarc_avg",
    steps=(step1, step2, step3)
)

oma_alg.add_clustering(clus)

Pali_ss.add_algorithms(oma_alg)
Pali_ss.run_by_name(OMA_METHOD)
Pali_ss[OMA_METHOD].run_all_clustering()

fig, ax = oma_alg.plot_stab_cluster("hierarc_avg")

clus_res = oma_alg.result.clustering_results["hierarc_avg"]

Fn_oma = np.asarray(clus_res.Fn, dtype=float)
Phi_oma = np.asarray(clus_res.Phi)

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
# SAVE RESULTS
# ============================================================
results = {
    "Fn_oma_Hz": Fn_oma,
    "Phi_oma": Phi_oma,
    "Phi_oma_real": outils.complex_to_normal_mode(Phi_oma),
    "FIM": FIM,
    "detFIM": detFIM,
    "Ed": Ed,
    "SSI_br": SSI_BR,
    "SSI_ordmin": SSI_ORDMIN,
    "SSI_ordmax": SSI_ORDMAX,
    "SSI_step": SSI_STEP,
}

outils.save_json_serialized(results, output_dir / f"{OMA_METHOD}_results.json")

###################################################################
# ANALYSIS OF SUBSETS of sensors
####################################################################

subsets = [[1, 2, 5], [0, 1, 2], [1, 2, 6], [0, 1, 5]]
figures_dir.mkdir(parents=True, exist_ok=True)

print(f"\n=== 2. COV-SSI computation using {OMA_METHOD} for sensor subsets ===")

all_results = {}

for subset_id, subset in enumerate(subsets, start=1):

    subset_name = "subset_" + "_".join(map(str, subset))
    subset_sensor_names = [selected_cols[i] for i in subset]

    print(f"\n--- Running {subset_name} ---")
    print("Sensor indices:", subset)
    print("Sensor names:", subset_sensor_names)

    data_subset = data[:, subset]

    Pali_ss = SingleSetup(data_subset, fs=fs)
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
        hc=False,
        sc=False,
        pre_cluster=False,
        pre_clus_typ="GMM"
    )

    step2 = Step2(
        algo="hierarc",
        linkage="average",
        dc="auto"
    )

    step3 = Step3(
        post_proc=["merge_similar", "fn_IQR", "min_size_pctg"],
        min_pctg=0.15,
        freqlim=SSI_FREQLIM,
        select="medoid"
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
    sensor_ids = [i.replace("Sensor", "S") for i in subset_sensor_names]
    ax.set_title(f"{OMA_METHOD} | Sensors: {', '.join(sensor_ids)} | Stabilization clustering")

    fig.savefig(
        figures_dir / f"{OMA_METHOD}_{subset_name}_stab_cluster.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    clus_res = oma_alg.result.clustering_results["hierarc_avg"]

    Fn_oma = np.asarray(clus_res.Fn, dtype=float)
    Phi_oma = np.asarray(clus_res.Phi)

    all_results[subset_name] = {
        "subset_id": subset_id,
        "subset_indices": subset,
        "subset_sensors": subset_sensor_names,
        "Fn_oma_Hz": Fn_oma,
        "Phi_oma": Phi_oma,
    }

    print(f"Fn_oma for {subset_name} [Hz]:")
    print(Fn_oma)

outils.save_json_serialized(all_results, output_dir / f"{OMA_METHOD}_subsets_results.json")