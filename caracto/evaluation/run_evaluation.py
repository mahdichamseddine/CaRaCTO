from pathlib import Path

import numpy as np

from caracto.calibration.caracto_calibration import RangeMethod
from caracto.calibration.run_calibration import run_calibration
from caracto.calibration.transformation import transformation_matrix
from caracto.cli import get_main_parser
from caracto.common import HD_1080, X0
from caracto.reconstruction.point_reconstruction import (
    calculate_mean_error,
    compute_point_3d,
)


def compute_errors(
    calibration_path: Path,
    x0: list[float],
    image_dimensions: tuple[int, int],
    range_method: RangeMethod | None = None,
    *,
    simulation_std: tuple[float, float, float] | None = None,  # r, theta, px
    subset: int | None = None,
) -> dict[str, tuple[float, float]]:
    calibration_setup, x_result = run_calibration(
        calibration_path,
        image_dimensions,
        x0,
        range_method=range_method,
        simulation_std=simulation_std,
        subset=subset,
    )
    h_result, h_result_inv = transformation_matrix(x_result)

    reprojected = []
    ground_truth = []
    for key in calibration_setup.measurement_keys:
        radar_range, radar_azimuth = calibration_setup.get_radar_measurements(key)
        _, camera_pixel = calibration_setup.get_camera_measurements(key)

        ground_truth.append(np.array(calibration_setup.optitrack_data[key]))
        reprojected.append(
            compute_point_3d(
                calibration_setup.camera_matrix_inv,
                h_result,
                h_result_inv,
                camera_pixel.T,
                radar_range,
                radar_azimuth,
            )
        )

    ground_truth = np.array(ground_truth)
    reprojected = np.array(reprojected).squeeze(1)
    errors = {}
    errors["3d"] = calculate_mean_error(ground_truth, reprojected)
    errors["2d"] = calculate_mean_error(ground_truth[:, 0:2], reprojected[:, 0:2])

    return errors


def single_run(
    calibration_path: Path,
    x0: list[float],
    image_dimensions: tuple[int, int],
    *,
    simulation_std: tuple[float, float, float] | None = None,  # r, theta, px
    subset: int | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    errors = {}
    errors["elnatour"] = compute_errors(
        calibration_path,
        x0,
        image_dimensions,
        None,
        simulation_std=simulation_std,
        subset=subset,
    )
    errors["caracto_radar"] = compute_errors(
        calibration_path,
        x0,
        image_dimensions,
        RangeMethod.RADAR,
        simulation_std=simulation_std,
        subset=subset,
    )
    errors["caracto_camera"] = compute_errors(
        calibration_path,
        x0,
        image_dimensions,
        RangeMethod.CAMERA,
        simulation_std=simulation_std,
        subset=subset,
    )

    return errors


def main() -> None:
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = args.dataset_path

    # Best initialization
    errors_0 = single_run(calibration_path, X0, HD_1080)
    print("Best initialization:")
    for key, value in errors_0.items():
        print(f"{key} 3D error: {value['3d'][0]:.3f} ± {value['3d'][1]:.3f}")
        print(f"{key} 2D error: {value['2d'][0]:.3f} ± {value['2d'][1]:.3f}")


if __name__ == "__main__":
    main()
