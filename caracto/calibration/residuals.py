import numpy as np


def compute_plane_residual(radar_3d: np.ndarray, radar_azimuth: float) -> float:
    res = radar_3d[0] * np.sin(radar_azimuth) - radar_3d[1] * np.cos(radar_azimuth)
    return float(res)


def compute_sphere_residual(radar_3d: np.ndarray, radar_range: float) -> float:
    res = np.sum(radar_3d**2) - radar_range**2
    return float(res)


def compute_z_residual(radar_3d: np.ndarray) -> float:
    res = np.abs(radar_3d[2])
    return float(res)
