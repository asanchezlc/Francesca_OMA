# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os

from scipy.signal import csd, find_peaks
from scipy.linalg import svd

from pyoma2.algorithms import FSDD, pLSCF
from pyoma2.setup import SingleSetup
from pyoma2.functions.gen import save_to_file, load_from_file


# ============================================================
# OUTPUT FOLDER
# ============================================================
output_dir = pathlib.Path("./plots")
output_dir.mkdir(exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_processed_data_from_setup(setup_obj, fallback_data=None):
    """
    Try to retrieve the processed data matrix from the pyOMA setup object.
    Falls back to the original data if no internal attribute is found.
    """
    candidate_attrs = ["data", "_data", "Y", "_Y"]
    for attr in candidate_attrs:
        if hasattr(setup_obj, attr):
            val = getattr(setup_obj, attr)
            if isinstance(val, np.ndarray):
                return val

    if fallback_data is not None:
        print("[WARNING] Processed data not found inside SingleSetup. Using fallback data.")
        return fallback_data

    raise AttributeError("Could not retrieve processed data from SingleSetup.")


def compute_cmif_from_data(data, fs, nperseg=2048, detrend="constant"):
    """
    Compute the cross-spectral density matrix and the CMIF
    (singular values of the spectral matrix at each frequency line).
    """
    n_samples, n_channels = data.shape
    nperseg = min(nperseg, n_samples)

    freqs = None
    Syy = None

    for i in range(n_channels):
        for j in range(n_channels):
            f_ij, Pxy_ij = csd(
                data[:, i],
                data[:, j],
                fs=fs,
                nperseg=nperseg,
                detrend=detrend,
            )

            if freqs is None:
                freqs = f_ij
                Syy = np.zeros((len(freqs), n_channels, n_channels), dtype=complex)

            Syy[:, i, j] = Pxy_ij

    # Remove DC
    mask = freqs > 0
    freqs = freqs[mask]
    Syy = Syy[mask, :, :]

    singular_values = np.zeros((len(freqs), n_channels), dtype=float)

    for k in range(len(freqs)):
        U, S, Vh = svd(Syy[k, :, :], full_matrices=True)
        singular_values[k, :] = np.real(S)

    return freqs, singular_values


def scipy_peak_picking_on_cmif(
    freqs,
    singular_values,
    n_modes=6,
    min_freq=1.0,
    max_freq=15.0,
    prominence_ratio=0.05,
    min_distance_hz=0.2,
):
    """
    Use scipy find_peaks on the first singular value of the CMIF.
    """
    sv1 = singular_values[:, 0]

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_sel = freqs[mask]
    sv1_sel = sv1[mask]

    if len(freqs_sel) == 0:
        raise ValueError("No frequencies available in the selected frequency range.")

    bg_level = float(np.median(sv1_sel)) + 1e-12
    prominence = bg_level * prominence_ratio

    dfreq = np.mean(np.diff(freqs_sel))
    min_distance_idx = max(1, int(min_distance_hz / dfreq))

    peaks, props = find_peaks(
        sv1_sel,
        prominence=prominence,
        distance=min_distance_idx,
    )

    if len(peaks) == 0:
        print("[WARNING] No peaks found with current settings. Using global maximum only.")
        peaks = np.array([int(np.argmax(sv1_sel))], dtype=int)

    peak_strength = sv1_sel[peaks]
    order = np.argsort(peak_strength)[::-1]
    peaks = peaks[order[: min(n_modes, len(peaks))]]

    selected_freqs = freqs_sel[peaks]
    selected_strengths = sv1_sel[peaks]

    return freqs_sel, sv1_sel, peaks, selected_freqs, selected_strengths


def save_peak_picking_results(freqs, strengths, output_path):
    df_out = pd.DataFrame({
        "selected_frequency_hz": freqs,
        "peak_strength": strengths
    })
    df_out.to_csv(output_path, index=False)
    print(f"Peak picking results saved to: {output_path}")


def plot_cmif_with_peaks(freqs, singular_values, selected_freqs, save_path=None, freqlim=(1, 15)):
    plt.figure(figsize=(10, 6))

    n_sv = singular_values.shape[1]
    for i in range(n_sv):
        lw = 2.5 if i == 0 else 1.2
        alpha = 1.0 if i == 0 else 0.6
        plt.plot(
            freqs,
            10 * np.log10(np.maximum(singular_values[:, i], 1e-16)),
            linewidth=lw,
            alpha=alpha,
            label=f"SV{i+1}"
        )

    if len(selected_freqs) > 0:
        sel_vals = []
        for f0 in selected_freqs:
            idx = np.argmin(np.abs(freqs - f0))
            sel_vals.append(10 * np.log10(max(singular_values[idx, 0], 1e-16)))
        plt.scatter(selected_freqs, sel_vals, color="red", s=60, label="SciPy selected peaks", zorder=5)

    plt.xlim(freqlim)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Singular value [dB]")
    plt.title("CMIF with SciPy Peak Picking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"CMIF with peaks saved to: {save_path}")

    plt.show()


def extract_phi_from_fsdd_result(fsdd_result):
    """
    Try to retrieve the mode shape matrix Phi from fsdd.result.
    This depends on the pyOMA2 version, so we try several possible names.
    """
    candidate_attrs = [
        "Phi",
        "phi",
        "mode_shapes",
        "ModeShapes",
        "V",
        "vec",
        "shapes",
    ]

    for attr in candidate_attrs:
        if hasattr(fsdd_result, attr):
            val = getattr(fsdd_result, attr)
            if isinstance(val, np.ndarray):
                print(f"[INFO] Mode shape matrix found in fsdd.result.{attr}")
                return val

    print("[WARNING] Could not automatically find Phi in fsdd.result.")
    print("[INFO] Available attributes in fsdd.result:")
    for k in vars(fsdd_result).keys():
        print("   ", k)

    return None


def save_modal_results(sensor_names, fsdd_result, output_dir):
    """
    Save modal frequencies, damping, and mode shape matrix Phi if available.
    """
    # Frequencies
    if hasattr(fsdd_result, "Fn"):
        freq_df = pd.DataFrame({
            "mode": [f"Mode_{i+1}" for i in range(len(fsdd_result.Fn))],
            "frequency_hz": fsdd_result.Fn
        })
        freq_path = output_dir / "10_fsdd_modal_frequencies.csv"
        freq_df.to_csv(freq_path, index=False)
        print(f"Saved modal frequencies to: {freq_path}")

    # Damping
    if hasattr(fsdd_result, "Xi"):
        xi_df = pd.DataFrame({
            "mode": [f"Mode_{i+1}" for i in range(len(fsdd_result.Xi))],
            "damping": fsdd_result.Xi
        })
        xi_path = output_dir / "11_fsdd_modal_damping.csv"
        xi_df.to_csv(xi_path, index=False)
        print(f"Saved modal damping to: {xi_path}")

    # Mode shape matrix
    Phi = extract_phi_from_fsdd_result(fsdd_result)
    if Phi is not None:
        # If Phi is complex, save both real and abs version
        # I modify here Francesca
        phi_complex = Phi
        # phi_abs = np.abs(Phi)

        # Try to orient Phi as (n_sensors, n_modes)
        if Phi.shape[0] == len(sensor_names):
            # phi_df = pd.DataFrame(
            #     phi_abs,
            #     index=sensor_names,
            #     columns=[f"Mode_{i+1}" for i in range(Phi.shape[1])]
            # )
            phi_df = pd.DataFrame(
                phi_complex,
                index=sensor_names,
                columns=[f"Mode_{i+1}" for i in range(Phi.shape[1])]
            )
        elif Phi.shape[1] == len(sensor_names):
            # phi_df = pd.DataFrame(
            #     phi_abs.T,
            #     index=sensor_names,
            #     columns=[f"Mode_{i+1}" for i in range(Phi.shape[0])]
            # )
            phi_df = pd.DataFrame(
                phi_complex.T,
                index=sensor_names,
                columns=[f"Mode_{i+1}" for i in range(Phi.shape[0])]
            )
        else:
            print("[WARNING] Phi shape is not compatible with sensor count.")
            print("Phi shape:", Phi.shape)
            return

        phi_df.index.name = "sensor"
        phi_path = output_dir / "12_fsdd_mode_shapes_phi_complex.csv"
        # phi_path = output_dir / "12_fsdd_mode_shapes_phi.csv"
        phi_df.to_csv(phi_path)
        print(f"Saved mode shape matrix Phi to: {phi_path}")


def plot_mode_shapes_from_phi(sensor_names, fsdd_result, output_dir):
    """
    Plot mode shapes if Phi is available inside fsdd.result.
    """
    Phi = extract_phi_from_fsdd_result(fsdd_result)
    if Phi is None:
        print("[WARNING] Skipping mode shape plot because Phi is not available.")
        return

    phi_abs = np.abs(Phi)

    if Phi.shape[0] == len(sensor_names):
        Phi_plot = phi_abs
    elif Phi.shape[1] == len(sensor_names):
        Phi_plot = phi_abs.T
    else:
        print("[WARNING] Skipping mode shape plot because Phi shape is incompatible.")
        print("Phi shape:", Phi.shape)
        return

    n_modes = Phi_plot.shape[1]
    fig, axes = plt.subplots(n_modes, 1, figsize=(10, 3 * n_modes), squeeze=False)

    for m in range(n_modes):
        ax = axes[m, 0]
        ax.bar(range(len(sensor_names)), Phi_plot[:, m])
        ax.set_title(f"FSDD Mode Shape {m+1}")
        ax.set_xticks(range(len(sensor_names)))
        ax.set_xticklabels(sensor_names, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "13_fsdd_mode_shapes_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Mode shape plot saved to: {out_path}")
    plt.show()

def compute_psd_matrix(df, sample_rate_hz, nperseg=None):
    signals = df.to_numpy(dtype=float)
    n_samples, n_sensors = signals.shape

    if nperseg is None:
        nperseg = min(1024, max(256, n_samples // 8))
    nperseg = min(nperseg, n_samples)

    freqs = None
    Syy = None

    for i in range(n_sensors):
        for j in range(n_sensors):
            f_ij, Pxy_ij = csd(
                signals[:, i],
                signals[:, j],
                fs=sample_rate_hz,
                nperseg=nperseg,
                detrend="constant",
            )

            if freqs is None:
                freqs = f_ij
                Syy = np.zeros((len(freqs), n_sensors, n_sensors), dtype=complex)

            Syy[:, i, j] = Pxy_ij

    return freqs, Syy


def run_fdd_manual(freqs, Syy, remove_dc=True):
    if remove_dc:
        mask = freqs > 0
        freqs = freqs[mask]
        Syy = Syy[mask, :, :]

    n_freqs, n_sensors, _ = Syy.shape
    singular_values = np.zeros((n_freqs, n_sensors), dtype=float)

    for k in range(n_freqs):
        _, S, _ = svd(Syy[k, :, :], full_matrices=True)
        singular_values[k, :] = np.real(S)

    return freqs, singular_values


def pick_modal_peaks(freqs, singular_values, n_modes=10, min_peak_prominence_ratio=0.05):
    s1 = singular_values[:, 0]
    bg_level = float(np.median(s1)) + 1e-12

    peaks, _ = find_peaks(
        s1,
        prominence=bg_level * min_peak_prominence_ratio,
    )

    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(s1))], dtype=int)

    peak_strength = s1[peaks]
    order = np.argsort(peak_strength)[::-1]
    selected = peaks[order[: min(n_modes, len(peaks))]]

    return selected


def plot_singular_values_manual(freqs, singular_values, selected_peak_indices=None, save_path=None):
    plt.figure(figsize=(10, 6))

    n_sv = singular_values.shape[1]

    for i in range(n_sv):
        lw = 2.5 if i == 0 else 1.5
        alpha = 1.0 if i == 0 else 0.6
        plt.plot(
            freqs,
            10 * np.log10(np.maximum(singular_values[:, i], 1e-16)),
            linewidth=lw,
            alpha=alpha,
            label=f"SV{i+1}"
        )

    if selected_peak_indices is not None:
        peak_freqs = freqs[selected_peak_indices]
        peak_vals = 10 * np.log10(
            np.maximum(singular_values[selected_peak_indices, 0], 1e-16)
        )
        plt.scatter(
            peak_freqs,
            peak_vals,
            marker="o",
            s=60,
            label="Selected peaks"
        )

    plt.xlim(1, 15)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Singular value [dB]")
    plt.title("FDD Singular Values - Manual Plot")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# ============================================================
# 1. LOAD DATA FROM EXCEL
# ============================================================
print("\n=== 1. LOADING DATA FROM EXCEL ===")

file_path = r"C:\Users\melig\Desktop\OMA\Roberto_signal_Puente_Circunv.xlsx"
fs = 100  # sampling frequency

try:
    df = pd.read_excel(file_path)
except:
    file_path = r"C:\Users\User\Documents\DOCTORADO_CODES\Francesca\Roberto_signal_Puente_Circunv.xlsx"
    df = pd.read_excel(file_path)

print("Columns found in the file:")
print(df.columns)

time_cols = [col for col in df.columns if "time" in str(col).lower()]
if len(time_cols) > 0:
    print(f"Time columns found and removed: {time_cols}")
    df = df.drop(columns=time_cols)

df = df.select_dtypes(include=[np.number])
df = df.dropna()

# Changes for Francesca
FREQ_LIM = (1.0, 15.0)
NPERSEG_FDD = 1024
N_MODES = 10
PROMINENCE_RATIO = 0.05
MIN_DISTANCE_HZ = 0.2

# ============================================================
# SELECT CHANNELS
# ============================================================
selected_cols = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5", "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10"]
df_sel = df[selected_cols].copy()

print("\nSelected sensors:")
print(df_sel.columns.tolist())

data = df_sel.values

print("Selected data shape:", data.shape)
print("Number of samples:", data.shape[0])
print("Number of selected channels:", data.shape[1])

Pali_ss = SingleSetup(data, fs=fs)


# ============================================================
# 2. INITIAL DATA PLOTS
# ============================================================
#print("\n=== 2. INITIAL DATA PLOTS ===")
#
#for i, sens_name in enumerate(df_sel.columns):
#    print(f"\nChannel info plot: {sens_name}")
#    figs, axs = Pali_ss.plot_ch_info(ch_idx=[i])
#
#    for j, fig in enumerate(figs):
#        fig.savefig(output_dir / f"02_ch_info_{sens_name}_{j+1}.png", dpi=300, bbox_inches="tight")
#        plt.show()
#        plt.close(fig)


# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n=== 3. PREPROCESSING ===")

Pali_ss.filter_data(Wn=0.2, order=8, btype="highpass")
print("High-pass filter applied: Wn=0.2 Hz, order=8")


# ============================================================
# 4. CHECK AFTER PREPROCESSING
# ============================================================
#print("\n=== 4. CHECK AFTER PREPROCESSING ===")
#
#for i, sens_name in enumerate(df_sel.columns):
#    print(f"\nPost-processing check: {sens_name}")
#    figs, axs = Pali_ss.plot_ch_info(ch_idx=[i])
#
#    for j, fig in enumerate(figs):
#        fig.savefig(output_dir / f"03_postproc_ch_info_{sens_name}_{j+1}.png", dpi=300, bbox_inches="tight")
#        plt.show()
#        plt.close(fig)


# ============================================================
# 5. DEFINE ALGORITHMS
# ============================================================
print("\n=== 5. DEFINING ALGORITHMS ===")

fsdd = FSDD(name="FSDD", nxseg=1024, method_SD="cor")
plscf = pLSCF(name="polymax", ordmax=30)

fsdd.run_params = FSDD.RunParamCls(
    nxseg=2048,
    method_SD="per",
    pov=0.5
)

Pali_ss.add_algorithms(fsdd, plscf)

print("Algorithms added:")
print("- FSDD")
print("- pLSCF (polymax)")

# ============================================================
# 5B. MANUAL FDD PLOT LIKE OLD CODE
# ============================================================
print("\n=== 5B. MANUAL FDD PLOT LIKE OLD CODE ===")

df_for_manual_fdd = pd.DataFrame(
    Pali_ss.data,
    columns=df_sel.columns
)

freqs_manual, Syy_manual = compute_psd_matrix(
    df_for_manual_fdd,
    sample_rate_hz=Pali_ss.fs,
    nperseg=NPERSEG_FDD
    # nperseg=None
)

freqs_manual, singular_values_manual = run_fdd_manual(
    freqs_manual,
    Syy_manual,
    remove_dc=True
)

selected_peak_indices_manual = pick_modal_peaks(
    freqs_manual,
    singular_values_manual,
    n_modes=N_MODES,
    min_peak_prominence_ratio=0.05
)

print("Manual FDD selected frequencies [Hz]:")
print(freqs_manual[selected_peak_indices_manual])

plot_singular_values_manual(
    freqs_manual,
    singular_values_manual,
    selected_peak_indices=selected_peak_indices_manual,
    save_path=output_dir / "manual_fdd_singular_values_1_15Hz.png"
)

# ============================================================
# 6. RUN FSDD
# ============================================================
print("\n=== 6. RUNNING FSDD ===")

Pali_ss.run_by_name("FSDD")
print("FSDD completed.")


# ============================================================
# 7. RUN pLSCF
# ============================================================
print("\n=== 7. RUNNING pLSCF ===")

Pali_ss.run_by_name("polymax")
print("pLSCF completed.")


# ============================================================
# 8. SAVE RESULTS IN VARIABLES
# ============================================================
print("\n=== 8. SAVING RESULTS ===")

fsdd_res = dict(fsdd.result)
plscf_res = dict(plscf.result) if plscf.result is not None else None

print("Results saved in variables.")


# ============================================================
# 9. FSDD PLOT (CMIF / SINGULAR VALUES)
# ============================================================
print("\n=== 9. FSDD PLOT ===")

fig, ax = fsdd.plot_CMIF(freqlim=(1, 15))
fig.savefig(output_dir / "04_fsdd_cmif_1_15Hz.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 10. SCIPY PEAK PICKING ON CMIF
# ============================================================
print("\n=== 10. SCIPY PEAK PICKING ON CMIF ===")

processed_data = get_processed_data_from_setup(Pali_ss, fallback_data=data)
freqs_cmif, singular_values = compute_cmif_from_data(
    processed_data,
    fs=Pali_ss.fs,
    nperseg=NPERSEG_FDD
)

# selected_peak_indices_manual = pick_modal_peaks(
#     freqs_manual,
#     singular_values_manual,
#     n_modes=10,
#     min_peak_prominence_ratio=0.05
# )

freqs_used, sv1_used, peak_idx_local, selected_freqs, selected_strengths = scipy_peak_picking_on_cmif(
    freqs_cmif,
    singular_values,
    # n_modes=6,
    n_modes=N_MODES,
    min_freq=1.0,
    max_freq=15.0,
    prominence_ratio=PROMINENCE_RATIO,
    # min_distance_hz=0.2,
)

print("Selected frequencies from SciPy peak picking:")
print(selected_freqs)

save_peak_picking_results(
    selected_freqs,
    selected_strengths,
    output_dir / "05_selected_frequencies_scipy.csv"
)

plot_cmif_with_peaks(
    freqs_cmif,
    singular_values,
    selected_freqs,
    save_path=output_dir / "06_cmif_with_scipy_peaks.png",
    freqlim=(1, 15)
)


# ============================================================
# 11. pLSCF STABILIZATION DIAGRAM
# ============================================================
print("\n=== 11. pLSCF STABILIZATION DIAGRAM ===")

fig, ax = plscf.plot_stab(freqlim=(1, 15), hide_poles=False)
fig.savefig(output_dir / "07_plscf_stab_1_15Hz.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 12. MODE EXTRACTION - FSDD USING SCIPY PEAKS
# ============================================================
print("\n=== 12. MODE EXTRACTION - FSDD USING SCIPY PEAKS ===")

sel_freq = list(selected_freqs)

Pali_ss.mpe("FSDD", sel_freq=sel_freq, MAClim=0.95)

print("Selected FSDD frequencies:")
print(fsdd.result.Fn)

if hasattr(fsdd.result, "Xi"):
    print("FSDD damping:")
    print(fsdd.result.Xi)


# ============================================================
# 13. SAVE MODAL RESULTS
# ============================================================
print("\n=== 13. SAVE MODAL RESULTS ===")

save_modal_results(
    sensor_names=df_sel.columns.tolist(),
    fsdd_result=fsdd.result,
    output_dir=output_dir,
)

plot_mode_shapes_from_phi(
    sensor_names=df_sel.columns.tolist(),
    fsdd_result=fsdd.result,
    output_dir=output_dir,
)


# ============================================================
# 14. RESULTS COMPARISON
# ============================================================
print("\n=== 14. RESULTS SUMMARY ===")

print("\n--- FSDD (SciPy peak-picked) ---")
print("Fn =", fsdd.result.Fn)
if hasattr(fsdd.result, "Xi"):
    print("Xi =", fsdd.result.Xi)


# ============================================================
# 15. OPTIONAL SAVE / RELOAD SETUP
# ============================================================
print("\n=== 15. OPTIONAL SAVE / RELOAD SETUP ===")

try:
    temp_file = pathlib.Path("./test.pkl")
    save_to_file(Pali_ss, temp_file)
    print("Setup saved to file.")

    pali2 = load_from_file(temp_file)
    print("Setup reloaded correctly.")

    os.remove(temp_file)
    print("Temporary file removed.")
except Exception as e:
    print("Save/reload failed:", e)



# ============================================================
# 16. ADDITIONAL: MODE SHAPE AND FIM MATRIX
# ============================================================
Phi = fsdd.result.Phi

FIM = Phi.conj().T @ Phi
detFIM = abs(np.linalg.det(FIM))
print("\n=== END OF SCRIPT ===")
