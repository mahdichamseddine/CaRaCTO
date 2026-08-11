import numpy as np


def cartesian_to_range_azimuth(target: np.ndarray) -> tuple[float, float]:
    range_m = np.sqrt(np.sum(target**2))
    azimuth = np.arctan2(target[1], target[0])
    return float(range_m), float(azimuth)


def spherical_to_cartesian(
    range_m: float,
    azimuth: float,
    elevation: float,
) -> np.ndarray:
    x = range_m * np.cos(azimuth) * np.cos(elevation)
    y = range_m * np.sin(azimuth) * np.cos(elevation)
    z = range_m * np.sin(elevation)
    return np.array([x, y, z])
