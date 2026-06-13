
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


def EFI(Phi, Phi_id, n_sensors):
    """
    Function Duties:
        Removes the least informative sensor from the set of
        sensors through the EFI algorithm
    Input:
        Phi: Matrix of mode shapes
        Phi_id: List of joint names
        n_sensors: Number of sensors to be maintained
    References:
        Enrique García-Macías' code-based implementation
        Kammer, D.C. Sensor placement for on-orbit modal identification
        and correlation of large space structures. J. Guid. Control Dyn.
        1991, 14, 251–259.
    """
    n_iter = len(Phi_id) - n_sensors
    detFIM, Ed_all = np.zeros(n_iter), list()
    deleted_channels = list()
    for i in range(n_iter):
        Q = np.dot(Phi.T, Phi)
        detFIM[i] = np.linalg.det(Q)
        Lambda, Psi = np.linalg.eig(Q)
        G = np.multiply(np.dot(Phi, Psi), np.dot(Phi, Psi))
        Fe = np.dot(G, np.linalg.inv(np.diag(Lambda)))
        Ed = np.sum(Fe, axis=1)
        # Ed = [np.dot(np.dot(np.dot(Phi, Psi), np.linalg.inv(np.diag(Lambda))), np.dot(Psi.T, Phi.T))[i, i] for i in range(np.shape(Phi)[0])]
        Ed_all.append(Ed.tolist())
        ranked_sensors_id = np.argsort(Ed)

        if Ed[ranked_sensors_id[0]] > 0.99:
            message = 'All sensors are informative'
            warnings.warn(message, UserWarning)

        channel_deleted_id = ranked_sensors_id[0]
        channel_deleted = Phi_id[channel_deleted_id]
        deleted_channels.append(channel_deleted)

        Phi = np.delete(Phi, channel_deleted_id, axis=0)
        Phi_id = np.delete(Phi_id, channel_deleted_id)

    EFI_results = dict()
    EFI_results['results'] = {'Phi': Phi.tolist(), 'Phi_id': Phi_id.tolist()}
    EFI_results['process'] = {'detFIM': detFIM.tolist(), 'Ed_all': Ed_all,
                              'deleted_channels': deleted_channels}

    return EFI_results


def plot_MAC_1(MAC, language='Spanish', modes_number=None,
               annot_kws=None, annot_kws_ticklabels=None,
               annot_kws_legend=None, vmin=10**-5, vmax=1):
    """
    modes_number: list for the labels.
        Example: if we want to label "Mode 1, Mode  4, Mode 7" then modes_number=[1, 4, 7]
    Example for annot_kws: annot_kws = {"fontsize": 10, "fontweight": "bold", "color": "black"}
    """

    n_modes = np.shape(MAC)[0]

    if language == 'Spanish':
        label = 'Modo'
    else:
        label = 'Mode'

    col = list()
    for kk in range(n_modes):
        if modes_number is not None:
            col.append(label + ' ' + str(modes_number[kk]))
        else:
            col.append(label + ' ' + str(kk+1))

    MAC = pd.DataFrame(MAC, columns=col, index=col)
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(MAC, cmap="jet", ax=ax, annot=True,
                fmt='.3f', annot_kws=annot_kws, vmin=vmin, vmax=vmax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    if annot_kws_ticklabels is not None:
        if 'fontsize' in annot_kws_ticklabels:
            for tick_label in ax.get_xticklabels():
                tick_label.set_fontsize(annot_kws_ticklabels["fontsize"])
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontsize(annot_kws_ticklabels["fontsize"])
        if 'fontweight' in annot_kws_ticklabels:
            for tick_label in ax.get_xticklabels():
                tick_label.set_fontweight(annot_kws_ticklabels["fontweight"])
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontweight(annot_kws_ticklabels["fontweight"])
        # Add other properties as needed

    if annot_kws_legend is not None:
        cbar = heatmap.collections[0].colorbar
        for t in cbar.ax.get_yticklabels():
            if 'fontsize' in annot_kws_legend:
                t.set_fontsize(annot_kws_legend["fontsize"])
            if 'fontweight' in annot_kws_legend:
                t.set_fontweight(annot_kws_legend["fontweight"])

    fig.tight_layout()

    return (fig, ax)

def MaC_nan(phi_X, phi_A):
    """
    MAC con soporte de NaNs: para cada par de modos (i, j) se omiten las filas
    donde phi_X[:, i] o phi_A[:, j] tengan NaN. Si tras omitir no quedan DOFs,
    el MAC(i, j) se devuelve como np.nan.

    Acepta entradas (n,) o (n, m). Devuelve escalar si ambos son 1D.
    """
    phi_X = np.asarray(phi_X)
    phi_A = np.asarray(phi_A)

    if phi_X.ndim == 1:
        phi_X = phi_X[:, np.newaxis]
    if phi_A.ndim == 1:
        phi_A = phi_A[:, np.newaxis]

    if phi_X.ndim > 2 or phi_A.ndim > 2:
        raise Exception(
            f'Mode shape matrices must have 1 or 2 dimensions (phi_X: {phi_X.ndim}, phi_A: {phi_A.ndim})'
        )

    if phi_X.shape[0] != phi_A.shape[0]:
        raise Exception(
            f'Mode shapes must have the same first dimension (phi_X: {phi_X.shape[0]}, phi_A: {phi_A.shape[0]})'
        )

    nX = phi_X.shape[1]
    nA = phi_A.shape[1]
    MAC = np.full((nX, nA), np.nan, dtype=float)

    for i in range(nX):
        x = phi_X[:, i]
        for j in range(nA):
            a = phi_A[:, j]

            valid = (~np.isnan(x)) & (~np.isnan(a))
            if not np.any(valid):
                continue  # deja NaN

            xv = x[valid]
            av = a[valid]

            num = np.abs(np.vdot(xv, av))**2          # vdot aplica conj al primero
            den = (np.vdot(xv, xv) * np.vdot(av, av))  # energías
            den = np.real(den)

            if den == 0:
                continue  # deja NaN (o podrías poner 0.0, según tu criterio)

            MAC[i, j] = np.real(num) / den

    if MAC.shape == (1, 1):
        return MAC[0, 0]
    return MAC


def save_json_serialized(obj, filepath, omit_keys=None) -> None:
    serial = serialize_dictionary_v2(obj, omit_keys=omit_keys)
    with open(filepath, 'w') as f:
        json.dump(serial, f, indent=2)


def load_json_serialized(filepath):
    with open(filepath, 'r') as f:
        return from_serializable(json.load(f))


def from_serializable(obj):
    """
    Deserializes objects encoded by `to_serializable`, restoring NumPy arrays,
    complex numbers, and nested structures to native Python types.

    Remark: enhanced version of deserilize_dict
    """
    if isinstance(obj, dict):
        if "__complex__" in obj:
            return complex(obj["real"], obj["imag"])
        elif "__complex_array__" in obj:
            return np.array(obj["real"]) + 1j * np.array(obj["imag"])
        else:
            return {k: from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Try converting to ndarray if list of numbers or complex values
        converted = [from_serializable(v) for v in obj]
        if all(isinstance(x, (float, int, complex, np.number)) for x in converted):
            return np.array(converted)
        elif all(isinstance(x, np.ndarray) for x in converted):
            try:
                return np.stack(converted)
            except Exception:
                return converted  # fallback: list of arrays
        return converted
    return obj


def serialize_dictionary_v2(test_dict, omit_keys=None):
    """
    Converts a dictionary into a JSON-serializable format, handling complex numbers,
    NumPy arrays, and other non-native JSON types.

    Parameters
    ----------
    test_dict : dict
        The dictionary containing results for one test (e.g., signal processing, FDD results).
    omit_keys : str or list of str, optional
        Key(s) to exclude from serialization (e.g., 'FDD' to avoid saving bulky internal data).

    Returns
    -------
    test_dict_serializable : dict
        A cleaned and fully JSON-compatible dictionary, suitable for writing to file.
    """
    if omit_keys is None:
        omit_keys = []
    elif isinstance(omit_keys, str):
        omit_keys = [omit_keys]

    test_dict_serializable = {}
    for key, value in test_dict.items():
        if key in omit_keys:
            continue
        test_dict_serializable[key] = to_serializable(value)

    return test_dict_serializable


def to_serializable(obj):
    """
    Converts NumPy arrays, complex numbers, and nested structures into
    JSON-compatible formats for safe serialization.
    """
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "__complex_array__": True,
                "real": obj.real.tolist(),
                "imag": obj.imag.tolist()
            }
        else:
            return obj.tolist()
    elif isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def _large_normal_mode_approx(mode, step, long):
    """Get normal mode approximation for large modes.
    [From EGM codes]

    In cases, where ``mode`` has ``n`` coordinates and
    ``n`` is large, this would result in a matrix ``U`` of
    size ``n x n``. To find eigenvalues of this non-sparse
    matrix is computationally expensive. The solution is to
    find the angle of the rotation for the vector - this is
    done using only every ``step`` element of ``mode``.
    The entire ``mode`` is then rotated, thus the full normal
    mode is obtained.

    Síntesis: buscar la orientación media de las componentes
    modales en el plano complejo para, a posteriori, rotar las
    componentes y tratar de alinearlas en el eje rea

    To ensure the influence of all the coordinates, a ``long``
    parameter can be used. Multiple angles of rotation are
    computed and then averaged.

    :param mode: a 2D mode shape or modal matrix ``(n_locations x n_modes)``
    :param step: int, every ``step`` elemenf of ``mode`` will be taken
        into account for angle of rotation calculation.
    :param long: bool, if True, the angle of rotation is computed
        iteratively for different starting positions (from 0 to ``step``), when
        every ``step`` element is taken into account.
    :return: normal mode or modal matrix of ``mode``.
    """
    if mode.ndim == 1:
        mode = mode[:, None]
    elif mode.ndim > 2:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim})')

    mode = mode / np.linalg.norm(mode, axis=0)[None, :]

    if long:
        step_long = step
    else:
        step_long = 1

    Alpha = []
    for i in range(step_long):
        mode_step = mode[i::step]
        mode_normal_step = complex_to_normal_mode(mode_step)

        v1 = np.concatenate(
            (np.real(mode_step)[:, :, None], np.imag(mode_step)[:, :, None]), axis=2)
        v2 = np.concatenate((np.real(mode_normal_step)[:, :, None], np.imag(
            mode_normal_step)[:, :, None]), axis=2)

        v1 /= np.linalg.norm(v1, axis=2)[:, :, None]
        v2 /= np.linalg.norm(v2, axis=2)[:, :, None]

        dot_product = np.array([np.matmul(np.transpose(v1[:, j, :, None], [0, 2, 1]),
                               v2[:, j, :, None]) for j in range(v1.shape[1])])
        angles = np.arccos(dot_product)

        alpha = np.mean(angles[:, :, 0, 0], axis=1)
        Alpha.append(alpha)

    alpha = np.mean(Alpha, axis=0)[None, :]

    mode_normal_full = np.real(mode)*np.cos(alpha) - \
        np.imag(mode)*np.sin(alpha)
    mode_normal_full /= np.linalg.norm(mode_normal_full, axis=0)[None, :]

    return mode_normal_full


def complex_to_normal_mode(mode, max_dof=50, long=True):
    """Transform a complex mode shape to normal mode shape.
    [From EGM codes]

    The real mode shape should have the maximum correlation with
    the original complex mode shape. The vector that is most correlated
    with the complex mode, is the real part of the complex mode when it is
    rotated so that the norm of its real part is maximized. [1]
    ``max_dof`` and ``long`` arguments are given for modes that have
    a large number of degrees of freedom. See ``_large_normal_mode_approx()``
    for more details.

    Literature:
        [1] Gladwell, H. Ahmadian GML, and F. Ismail.
            "Extracting Real Modes from Complex Measured Modes."
            (avaliable in 'doc' folder)

    :param mode: np.ndarray, a mode shape to be transformed. Can contain a single
        mode shape or a modal matrix `(n_locations, n_modes)`.
    :param max_dof: int, maximum number of degrees of freedom that can be in
        a mode shape. If larger, ``_large_normal_mode_approx()`` function
        is called. Defaults to 50.
    :param long: bool, If True, the start in stepping itartion is altered, the
        angles of rotation are averaged (more in ``_large_normal_mode_approx()``).
        This is needed only when ``max_dof`` is exceeded. The normal modes are
        more closely related to the ones computed with an entire matrix. Defaults to True.
    :return: normal mode shape
    """
    if mode.ndim == 1:
        mode = mode[None, :, None]
    elif mode.ndim == 2:
        mode = mode.T[:, :, None]
    else:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim}).')

    # if mode.shape[1] > max_dof   --> Computationally expensive
    if mode.shape[1] > max_dof:
        return _large_normal_mode_approx(mode[:, :, 0].T, step=int(np.ceil(mode.shape[1] / max_dof)) + 1, long=long)

    # 1. Normalize modes so that norm == 1.0
    _norm = np.linalg.norm(mode, axis=1)[:, None, :]
    mode = mode / _norm

    # 2. Obtain U matrix
    mode_T = np.transpose(mode, [0, 2, 1])
    U = np.matmul(np.real(mode), np.real(mode_T)) + \
        np.matmul(np.imag(mode), np.imag(mode_T))

    # Modification to operate without nan values (otherwise np.linalg.eig raise error)
    nan_mode = np.all(np.isnan(U), axis=(1, 2))
    nan_index = np.where(nan_mode)[0]
    if nan_index.size > 0:
        not_nan = [not (i) for i in nan_mode]
        U_copy = U[not_nan, :, :]
    else:
        U_copy = U

    # 3. Obtain eigenvectors & eigenvalues and choose eigenvector associated to max eigenvalue
    val, vec = np.linalg.eig(U_copy)
    # modification to get as a result mode=0 for nan values [spureous modes]
    if nan_index.size > 0:
        val_aux = np.empty((np.shape(val)[0]+len(nan_index), np.shape(val)[1]))
        vec_aux = np.empty(
            (np.shape(U_copy)[0]+len(nan_index), np.shape(U_copy)[1], np.shape(U_copy)[2]))
        val_aux[not_nan, :] = val
        vec_aux[not_nan, :, :] = vec
        for j in nan_index:
            val_aux[not_nan, :] = np.zeros((np.shape(U_copy)[1]))
            vec_aux[j, :, :] = np.zeros(
                (np.shape(U_copy)[1], np.shape(U_copy)[2]))
        i = np.argmax(np.real(val_aux), axis=1)
        normal_mode = np.real([v[:, _] for v, _ in zip(vec_aux, i)]).T
    else:  # in normal cases we are here
        i = np.argmax(np.real(val), axis=1)
        normal_mode = np.real([v[:, _] for v, _ in zip(vec, i)]).T

    return normal_mode


def _sval_to_matrix(S_val, freq):
    """
    Converts singular values (S_val) into
    (n_frequencies, n_singular_values) matrix form.
    """
    S = np.asarray(S_val)
    freq = np.asarray(freq).ravel()
    n_freq = freq.size

    S = np.squeeze(S)

    freq_axes = [ax for ax, size in enumerate(S.shape) if size == n_freq]

    S = np.moveaxis(S, freq_axes[0], 0)

    if S.ndim == 2:
        return S

    if S.ndim == 3 and S.shape[1] == S.shape[2]:
        return np.diagonal(S, axis1=1, axis2=2)

    return S.reshape(n_freq, -1)
