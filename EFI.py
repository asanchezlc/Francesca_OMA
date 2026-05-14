

import numpy as np
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


# Example
Phi = np.array([
    [ 0.12,  0.35,  0.08],
    [ 0.18,  0.31,  0.15],
    [ 0.27,  0.22,  0.24],
    [ 0.36,  0.10,  0.32],
    [ 0.41, -0.05,  0.38],
    [ 0.33, -0.18,  0.29],
    [ 0.21, -0.27,  0.17],
    [ 0.09, -0.33,  0.05],
    [-0.05, -0.29, -0.12],
    [-0.16, -0.20, -0.25]
])

Phi_id = [
    "Sensor_01",
    "Sensor_02",
    "Sensor_03",
    "Sensor_04",
    "Sensor_05",
    "Sensor_06",
    "Sensor_07",
    "Sensor_08",
    "Sensor_09",
    "Sensor_10"
]

n_sensors = 3

EFI_results = EFI(Phi, Phi_id, n_sensors)

print("Selected sensors:")
print(EFI_results["results"]["Phi_id"])

print("\nReduced Phi:")
print(np.array(EFI_results["results"]["Phi"]))

print("\nDeleted sensors:")
print([str(i) for i in EFI_results["process"]["deleted_channels"]])

print("\nFIM determinant at each iteration:")
print(EFI_results["process"]["detFIM"])