from pathlib import Path

import numpy as np

from caracto.calibration.calibration_setup import CalibrationSetup
from caracto.dataset.camera_matrix import undistort_and_crop
from caracto.dataset.file_readers import read_file


def load_sample_data(
    calibration_path: Path, calibration_setup: CalibrationSetup, sample_name: str
) -> tuple[float, float, np.ndarray]:
    radar_path = calibration_path / "RadarData"
    radar_file = radar_path / (sample_name + ".pickle")
    measurement_data = read_file(radar_file)

    radar_range, radar_azimuth = calibration_setup.get_radar_measurements(sample_name)
    input_image = undistort_and_crop(
        measurement_data["Camera"][0],
        calibration_setup.old_camera_matrix,
        calibration_setup.dist_coeff,
        calibration_setup.camera_matrix,
        calibration_setup.roi,
    )

    return radar_range, radar_azimuth, input_image
