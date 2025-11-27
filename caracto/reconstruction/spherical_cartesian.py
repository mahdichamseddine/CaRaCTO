import numpy as np


def cartesian_to_range_azimuth(target: np.ndarray) -> tuple[float, float]:
    range = np.sqrt(np.sum(target**2))
    azimuth = np.arctan2(target[1], target[0])
    return float(range), float(azimuth)


def spherical_to_cartesian(
    range: float, azimuth: float, elevation: float
) -> np.ndarray:
    x = range * np.cos(azimuth) * np.cos(elevation)
    y = range * np.sin(azimuth) * np.cos(elevation)
    z = range * np.sin(elevation)
    return np.array([x, y, z])
