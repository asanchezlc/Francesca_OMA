
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os
import json
import helpers.outils as outils

from scipy.signal import find_peaks, peak_prominences, peak_widths

from pyoma2.algorithms.fdd import FDD, EFDD, FSDD
from pyoma2.setup import SingleSetup






# ============================================================
# PARAMETERS
# ============================================================
output_dir = pathlib.Path("./results")
figures_dir = output_dir / "plots"
output_dir.mkdir(exist_ok=True)

OMA_METHOD = "EFDD"  # "FDD", "EFDD", or "FSDD"; Best results with EFDD
OMA_METHOD = OMA_METHOD.upper()

# Parameter for Freq. Domain Analysis
NPERSEG_FDD = 4096  # 4096 results in a good balance between freq. res. and averaging

# Parameters for peak picking
PP_FREQLIM = (2.0, 10.0)     # Frequency range [Hz]; use None to analyze the full range
PP_USE_DB = True             # True: perform peak picking on 20*log10(SV); False: use linear scale
PP_EPS = 1e-30               # Prevents log10(0)
PP_MIN_HEIGHT = None         # Minimum peak height. In dB if PP_USE_DB=True
PP_MIN_PROMINENCE = 1.0      # Minimum peak prominence. In dB if PP_USE_DB=True
PP_MIN_DISTANCE_HZ = 0.2    # Minimum distance between peaks [Hz]
PP_MIN_WIDTH_HZ = 0.03       # Minimum peak width [Hz]
PP_WLEN_HZ = None            # Window length used to compute peak prominence [Hz]
PP_REL_HEIGHT = 0.5          # Relative height level used to compute peak width


# ============================================================
# LOAD DATA
# ============================================================
print("\n=== 1. LOADING DATA FROM EXCEL ===")

file_path = r"C:\Users\melig\Desktop\OMA\Roberto_signal_Puente_Circunv.xlsx"
fs = 100  # sampling frequency

target_freqs = np.array([
    2.87,
    3.76,
    4.53,
    5.05,
    5.31,
    6.14,
    6.42,
    6.81,
    7.24,
    8.31,
    9.51
])

try:
    df = pd.read_excel(file_path)
except:
    file_path = r"C:\Users\User\Documents\DOCTORADO_CODES\Francesca\Roberto_signal_Puente_Circunv.xlsx"
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
print(f"\n=== 2. SV Computation using {OMA_METHOD} ===")

Pali_ss = SingleSetup(data, fs=fs)
Pali_ss.filter_data(Wn=0.2, order=8, btype="highpass")
OMA_ALGORITHMS = {
    "FDD": FDD,
    "EFDD": EFDD,
    "FSDD": FSDD,
}

if OMA_METHOD not in OMA_ALGORITHMS:
    raise ValueError("OMA_METHOD must be 'FDD', 'EFDD', or 'FSDD'.")

oma_cls = OMA_ALGORITHMS[OMA_METHOD]
oma_alg = oma_cls(name=OMA_METHOD, nxseg=1024, method_SD="cor")
oma_alg.run_params = oma_alg.RunParamCls(
    nxseg=NPERSEG_FDD,
    method_SD="per",
    pov=0.5
)
Pali_ss.add_algorithms(oma_alg)
Pali_ss.run_by_name(OMA_METHOD)

print("\n=== 3. Selecting peaks ===")
freq = np.asarray(oma_alg.result.freq).ravel()
S_mat = outils._sval_to_matrix(oma_alg.result.S_val, freq)

if PP_FREQLIM is not None:
    fmin, fmax = PP_FREQLIM
    band_mask = (freq >= fmin) & (freq <= fmax)
else:
    band_mask = np.ones_like(freq, dtype=bool)

freq_pp = freq[band_mask]
S_pp = S_mat[band_mask, :]
dfreq = np.median(np.diff(freq_pp))

distance_pts = None if PP_MIN_DISTANCE_HZ is None else max(1, int(np.ceil(PP_MIN_DISTANCE_HZ / dfreq)))
width_pts = None if PP_MIN_WIDTH_HZ is None else max(1, PP_MIN_WIDTH_HZ / dfreq)
wlen_pts = None if PP_WLEN_HZ is None else max(1, int(np.ceil(PP_WLEN_HZ / dfreq)))

peak_rows = []
sv_idx = 0  # we start by first singular value

y_linear = S_pp[:, sv_idx]

if PP_USE_DB:
    y = 20.0 * np.log10(np.maximum(y_linear, PP_EPS))
    y_label = f"SV{sv_idx + 1} [dB]"
else:
    y = y_linear
    y_label = f"SV{sv_idx + 1} [-]"

peaks, props = find_peaks(
    y,
    height=PP_MIN_HEIGHT,
    prominence=PP_MIN_PROMINENCE,
    distance=distance_pts,
    width=width_pts,
    wlen=wlen_pts,
    rel_height=PP_REL_HEIGHT
)

if peaks.size > 0:
    prominences, left_bases, right_bases = peak_prominences(y, peaks, wlen=wlen_pts)
    widths, width_heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=PP_REL_HEIGHT)

    left_freqs = np.interp(left_ips, np.arange(freq_pp.size), freq_pp)
    right_freqs = np.interp(right_ips, np.arange(freq_pp.size), freq_pp)
    widths_hz = right_freqs - left_freqs

    for i, pk in enumerate(peaks):
        peak_rows.append({
            "SV": sv_idx + 1,
            "peak_index": int(pk),
            "frequency_Hz": freq_pp[pk],
            "amplitude": y[pk],
            "prominence": prominences[i],
            "width_Hz": widths_hz[i],
            "left_freq_Hz": left_freqs[i],
            "right_freq_Hz": right_freqs[i],
        })

plt.figure(figsize=(10, 4))
plt.plot(freq_pp, y, label=y_label)
plt.plot(freq_pp[peaks], y[peaks], "x", label="Picked peaks")
plt.xlabel("Frequency [Hz]")
plt.ylabel(y_label)
plt.title(f"Peak picking - Singular Value {sv_idx + 1}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(figures_dir / f"{OMA_METHOD}_peak_picking.png")

# SELECTING PEAKS
peaks_df = pd.DataFrame(peak_rows)

# MPE constants
MPE_MACLIM = 0.95       # MAC threshold used by FSDD during modal extraction
MPE_DF1 = 0.10          # Frequency resolution around each selected peak [Hz]
MPE_DF2 = 1.00          # Secondary frequency resolution/window parameter [Hz]
MPE_CM = 1              # Number of closely spaced modes
MPE_SPPK = 3            # Number of points used around the spectral peak
MPE_NPMAX = 2          # Maximum number of points for the SDOF bell identification

if peaks_df.empty:
    raise ValueError("No peaks were found. Modal extraction cannot be performed.")

sel_freq = peaks_df["frequency_Hz"].to_numpy()
sel_freq = np.sort(sel_freq)

print(f"\nSelected frequencies for {OMA_METHOD} modal extraction [Hz]:")
print(sel_freq)

MPE_DF = 0.10          # Frequency bandwidth around each selected peak [Hz], only for FDD
if OMA_METHOD == "FDD":
    Pali_ss.mpe(
        "FDD",
        sel_freq=sel_freq.tolist(),
        DF=MPE_DF
    )
elif OMA_METHOD in ["EFDD", "FSDD"]:
    Pali_ss.mpe(
        OMA_METHOD,
        sel_freq=sel_freq.tolist(),
        DF1=MPE_DF1,
        DF2=MPE_DF2,
        cm=MPE_CM,
        MAClim=MPE_MACLIM,
        sppk=MPE_SPPK,
        npmax=MPE_NPMAX
    )

Fn_oma = np.asarray(oma_alg.result.Fn, dtype=float)
Phi_oma = np.asarray(oma_alg.result.Phi)

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
    "Ed": Ed
}

outils.save_json_serialized(results, output_dir / f"{OMA_METHOD}_results.json")
