
import helpers.outils as outils
import os

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _extract_modal_data(data, modal_key="Phi_oma_real"):
    """
    Extrae frecuencias y matriz modal garantizando formato:
        Phi.shape = (n_sensors, n_modes)
        Fn.shape  = (n_modes,)
    """
    Fn = np.asarray(data["Fn_oma_Hz"], dtype=float).ravel()
    Phi = np.asarray(data[modal_key])

    if Phi.ndim != 2:
        raise ValueError(f"{modal_key} debe ser 2D. Shape recibido: {Phi.shape}")

    if Phi.shape[1] != Fn.size:
        if Phi.shape[0] == Fn.size:
            Phi = Phi.T
        else:
            raise ValueError(
                f"Incompatibilidad entre {modal_key}.shape={Phi.shape} "
                f"y len(Fn_oma_Hz)={Fn.size}"
            )

    return Fn, Phi


def _get_sensor_names(data, n_sensors, case_name="case"):
    """
    Devuelve sensor_names si existen. Si no existen, infiere Sensor1...SensorN.
    """
    if "sensor_names" in data:
        sensor_names = list(data["sensor_names"])

        if len(sensor_names) != n_sensors:
            raise ValueError(
                f"{case_name}: len(sensor_names)={len(sensor_names)} "
                f"pero la matriz modal tiene {n_sensors} sensores."
            )

        return sensor_names

    return [f"Sensor{i+1}" for i in range(n_sensors)]


def _expand_to_reference_sensors(Phi, sensor_names, ref_sensor_names):
    """
    Proyecta una matriz modal de un subset sobre la base completa de sensores
    de la referencia. Los sensores no presentes quedan como NaN.

    Salida:
        Phi_grid.shape = (n_ref_sensors, n_modes_subset)
    """
    dtype = np.result_type(Phi.dtype, float)
    Phi_grid = np.full((len(ref_sensor_names), Phi.shape[1]), np.nan, dtype=dtype)

    ref_index = {name: i for i, name in enumerate(ref_sensor_names)}

    for local_i, sensor in enumerate(sensor_names):
        if sensor not in ref_index:
            raise ValueError(f"El sensor {sensor} no existe en la referencia.")

        global_i = ref_index[sensor]
        Phi_grid[global_i, :] = Phi[local_i, :]

    return Phi_grid


def load_modal_database(folder_path, reference_file, modal_key="Phi_oma_real"):
    """
    Carga todos los resultados EFDD de los subsets y la referencia.

    Requiere que ya tengas disponibles:
        - outils.load_json_serialized
        - MaC_nan
    """
    ref_data = outils.load_json_serialized(reference_file)
    Fn_ref, Phi_ref = _extract_modal_data(ref_data, modal_key=modal_key)

    ref_sensor_names = _get_sensor_names(
        ref_data,
        n_sensors=Phi_ref.shape[0],
        case_name="reference"
    )

    modal_db = {
        "modal_key": modal_key,
        "reference": {
            "file": reference_file,
            "Fn_oma_Hz": Fn_ref,
            "Phi": Phi_ref,
            "sensor_names": ref_sensor_names,
            "AutoMAC": outils.MaC_nan(Phi_ref, Phi_ref),
            "raw": ref_data,
        },
        "tests": {}
    }

    for folder in sorted(os.listdir(folder_path)):
        case_dir = os.path.join(folder_path, folder)

        if not os.path.isdir(case_dir):
            continue

        file = os.path.join(case_dir, folder, f"{folder}_EFDD_results.json")

        if not os.path.isfile(file):
            file = os.path.join(case_dir, f"{folder}_EFDD_results.json")

        if not os.path.isfile(file):
            print(f"Archivo no encontrado para {folder}: {file}")
            continue

        data = outils.load_json_serialized(file)

        case_name = data.get("run_name", folder)

        Fn, Phi = _extract_modal_data(data, modal_key=modal_key)
        sensor_names = _get_sensor_names(
            data,
            n_sensors=Phi.shape[0],
            case_name=case_name
        )

        Phi_on_ref_grid = _expand_to_reference_sensors(
            Phi=Phi,
            sensor_names=sensor_names,
            ref_sensor_names=ref_sensor_names
        )

        MAC_vs_reference = outils.MaC_nan(Phi_ref, Phi_on_ref_grid)
        AutoMAC = outils.MaC_nan(Phi, Phi)

        modal_db["tests"][case_name] = {
            "folder": folder,
            "file": file,
            "Fn_oma_Hz": Fn,
            "Phi": Phi,
            "Phi_on_reference_sensors": Phi_on_ref_grid,
            "sensor_names": sensor_names,
            "MAC_vs_reference": MAC_vs_reference,
            "AutoMAC": AutoMAC,
            "raw": data,
        }

    return modal_db


def plot_frequencies_by_test(modal_db, include_reference=True):
    rows = []

    if include_reference:
        rows.append(("reference", modal_db["reference"]["Fn_oma_Hz"]))

    for test_name, case in modal_db["tests"].items():
        rows.append((test_name, case["Fn_oma_Hz"]))

    fig_height = max(4, 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ref_freqs = modal_db["reference"]["Fn_oma_Hz"]

    for f_ref in ref_freqs:
        ax.axvline(
            f_ref,
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            color="black",
            zorder=0
        )

    for y, (name, freqs) in enumerate(rows):
        ax.scatter(freqs, np.full_like(freqs, y, dtype=float), s=28)

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([name for name, _ in rows])
    ax.invert_yaxis()

    ax.set_xlabel("Identified frequency [Hz]")
    ax.set_ylabel("Test")
    ax.set_title("Identified frequencies per test")
    ax.grid(True, axis="x", alpha=0.35)

    fig.tight_layout()
    return fig, ax


def plot_MAC_vs_reference(modal_db, test_name, vmin=0.0, vmax=1.0):
    case = modal_db["tests"][test_name]

    MAC = case["MAC_vs_reference"]

    Fn_ref = modal_db["reference"]["Fn_oma_Hz"]
    Fn_test = case["Fn_oma_Hz"]

    ylabels = [
        f"Ref M{i+1}\n{fn:.3f} Hz"
        for i, fn in enumerate(Fn_ref)
    ]

    xlabels = [
        f"{test_name} M{j+1}\n{fn:.3f} Hz"
        for j, fn in enumerate(Fn_test)
    ]

    MAC_df = pd.DataFrame(MAC, index=ylabels, columns=xlabels)

    fig_width = max(8, 0.65 * MAC.shape[1])
    fig_height = max(5, 0.45 * MAC.shape[0])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        MAC_df,
        cmap="jet",
        ax=ax,
        annot=True,
        fmt=".3f",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "MAC"}
    )

    ax.set_xlabel("Modos identificados en el test")
    ax.set_ylabel("Modos de referencia")
    ax.set_title(f"MAC vs referencia - {test_name}")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    return fig, ax


def plot_frequencies_by_test_2(
    modal_db,
    include_reference=True,
    mac_vmin=None,
    mac_vmax=None,
    figsize_width=16,
    scatter_size=42,
    edgecolor="black",
    linewidth=0.6,
    add_reference_lines=True,
    add_colorbar=True,
):
    """
    Plotea las frecuencias identificadas por test.

    Cada punto de los tests se colorea en escala de grises según el MAC entre:
        - el modo identificado del test;
        - el modo de referencia cuya frecuencia es más cercana.

    Convención:
        MAC bajo  -> claro
        MAC alto  -> oscuro

    Parámetros
    ----------
    modal_db : dict
        Diccionario generado con load_modal_database.

    include_reference : bool
        Si True, incluye la fila de referencia.

    mac_vmin, mac_vmax : float or None
        Límites de normalización de la escala de grises.
        Si None, se calculan automáticamente usando los valores MAC realmente usados.

    figsize_width : float
        Ancho de la figura.

    scatter_size : float
        Tamaño de los puntos.

    edgecolor : str
        Color del borde de los puntos.

    linewidth : float
        Grosor del borde de los puntos.

    add_reference_lines : bool
        Si True, añade líneas verticales discontinuas en las frecuencias de referencia.

    add_colorbar : bool
        Si True, añade barra de color MAC.

    Returns
    -------
    fig, ax
    """

    ref_freqs = np.asarray(modal_db["reference"]["Fn_oma_Hz"], dtype=float).ravel()

    rows = []
    mac_values_used = []

    if include_reference:
        rows.append({
            "name": "reference",
            "freqs": ref_freqs,
            "mac_colors": np.ones_like(ref_freqs, dtype=float),
            "is_reference": True,
        })

    for test_name, case in modal_db["tests"].items():
        freqs = np.asarray(case["Fn_oma_Hz"], dtype=float).ravel()
        MAC_vs_reference = np.asarray(case["MAC_vs_reference"], dtype=float)

        mac_colors = np.full(freqs.shape, np.nan, dtype=float)
        nearest_ref_modes = np.full(freqs.shape, -1, dtype=int)

        for j, f_test in enumerate(freqs):
            i_ref = int(np.argmin(np.abs(ref_freqs - f_test)))
            nearest_ref_modes[j] = i_ref
            mac_colors[j] = MAC_vs_reference[i_ref, j]

        mac_values_used.extend(mac_colors[np.isfinite(mac_colors)])

        rows.append({
            "name": test_name,
            "freqs": freqs,
            "mac_colors": mac_colors,
            "nearest_ref_modes": nearest_ref_modes,
            "is_reference": False,
        })

    mac_values_used = np.asarray(mac_values_used, dtype=float)

    if mac_vmin is None:
        mac_vmin = np.nanmin(mac_values_used)

    if mac_vmax is None:
        mac_vmax = np.nanmax(mac_values_used)

    if mac_vmin == mac_vmax:
        mac_vmin = max(0.0, mac_vmin - 0.05)
        mac_vmax = min(1.0, mac_vmax + 0.05)

    norm = plt.Normalize(vmin=mac_vmin, vmax=mac_vmax)
    cmap = plt.cm.coolwarm
    # cmap = plt.cm.Greys

    fig_height = max(8, 0.7 * len(rows))
    fig, ax = plt.subplots(figsize=(figsize_width, fig_height))

    if add_reference_lines:
        for ii, f_ref in enumerate(ref_freqs):
            ax.axvline(
                f_ref,
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
                color="black",
                zorder=0,
                label="Ref. frequency" if ii == 0 else None,
            )

    for y, row in enumerate(rows):
        freqs = row["freqs"]
        mac_colors = row["mac_colors"]

        ax.scatter(
            freqs,
            np.full_like(freqs, y, dtype=float),
            c=mac_colors,
            cmap=cmap,
            norm=norm,
            s=scatter_size,
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=3,
        )

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["name"] for row in rows])
    ax.invert_yaxis()

    ax.set_xlabel("FIdentified frequency [Hz]")
    ax.set_ylabel("Test")
    ax.set_title("Identified frequencies per test (color = MAC vs closest reference mode)")
    ax.grid(True, axis="x", alpha=0.35)

    if add_colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("MAC with closest mode from reference")

    if add_reference_lines:
        ax.legend(loc="best")

    fig.tight_layout()

    return fig, ax


folder_path = r"C:\Users\User\Documents\DOCTORADO_CODES\Francesca\OMA_subset_runs"
fig_path = r"C:\Users\User\Documents\DOCTORADO_CODES\Francesca\plots"
folders = os.listdir(folder_path)

reference_file = os.path.join(
    os.path.dirname(folder_path),
    "results", "benchmark_full_10sensors",
    "benchmark_full_10sensors_EFDD_results.json"
)

modal_db = load_modal_database(
    folder_path=folder_path,
    reference_file=reference_file,
    modal_key="Phi_oma_real"
)

for folder in folders:
    file = os.path.join(folder_path, folder, f"{folder}_EFDD_results.json")
    data = outils.load_json_serialized(file)

fig, ax = plot_frequencies_by_test(modal_db, include_reference=True)
plt.show()

fig, ax = plot_frequencies_by_test_2(modal_db)
plt.show()
fig.savefig(os.path.join(fig_path, "frequencies_by_test.png"))

# test_name = "subset_1_best"
# for test_name in modal_db["tests"].keys():
#     fig, ax = plot_MAC_vs_reference(modal_db, test_name)
