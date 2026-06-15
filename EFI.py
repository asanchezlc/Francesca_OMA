

import numpy as np
import warnings




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