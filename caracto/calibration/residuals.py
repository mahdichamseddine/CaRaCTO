"""Individual residual terms shared by the calibration baselines."""

import numpy as np


def compute_plane_residual(radar_3d: np.ndarray, radar_azimuth: float) -> float:
    """Deviation of radar_3d from the vertical plane at radar_azimuth."""
    res = radar_3d[0] * np.sin(radar_azimuth) - radar_3d[1] * np.cos(radar_azimuth)
    return float(res)


def compute_sphere_residual(radar_3d: np.ndarray, radar_range: float) -> float:
    """Deviation of radar_3d's squared norm from radar_range squared."""
    res = np.sum(radar_3d**2) - radar_range**2
    return float(res)


def compute_z_residual(radar_3d: np.ndarray) -> float:
    """Deviation of radar_3d from the ground plane (z = 0)."""
    res = np.abs(radar_3d[2])
    return float(res)
