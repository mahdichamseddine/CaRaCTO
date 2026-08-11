"""Loads one position's camera/radar sample, undistorted and reconstruction-ready."""

from pathlib import Path

import numpy as np

from caracto.calibration.calibration_setup import CalibrationSetup
from caracto.dataset.camera_matrix import undistort_and_crop


def load_sample_data(
    calibration_path: Path,  # noqa: ARG001 (kept for call-site signature stability)
    calibration_setup: CalibrationSetup,
    sample_name: str,
) -> tuple[float, float, np.ndarray]:
    """Return sample_name's radar range/azimuth and undistorted camera frame."""
    camera_frame = calibration_setup.dataset.load_camera_frame(sample_name, 0)

    radar_range, radar_azimuth = calibration_setup.get_radar_measurements(sample_name)
    input_image = undistort_and_crop(
        camera_frame,
        calibration_setup.old_camera_matrix,
        calibration_setup.dist_coeff,
        calibration_setup.camera_matrix,
        calibration_setup.roi,
    )

    return radar_range, radar_azimuth, input_image
