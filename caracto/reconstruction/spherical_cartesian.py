"""Conversion between Cartesian points and range/azimuth/elevation."""

import numpy as np


def cartesian_to_range_azimuth(target: np.ndarray) -> tuple[float, float]:
    """Return (range, azimuth) for a Cartesian target point."""
    range_m = np.sqrt(np.sum(target**2))
    azimuth = np.arctan2(target[1], target[0])
    return float(range_m), float(azimuth)


def spherical_to_cartesian(
    range_m: float,
    azimuth: float,
    elevation: float,
) -> np.ndarray:
    """Return the Cartesian point at the given range/azimuth/elevation."""
    x = range_m * np.cos(azimuth) * np.cos(elevation)
    y = range_m * np.sin(azimuth) * np.cos(elevation)
    z = range_m * np.sin(elevation)
    return np.array([x, y, z])
