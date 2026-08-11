import numpy as np

from caracto.calibration.caracto_calibration import RangeMethod
from caracto.calibration.run_calibration import run_calibration
from caracto.calibration.transformation import transformation_matrix
from caracto.cli import get_main_parser, resolve_dataset_path
from caracto.common import HD_1080, X0
from caracto.reconstruction.spherical_cartesian import cartesian_to_range_azimuth


def compute_point_3d(
    camera_matrix_inv: np.ndarray,
    h_mat: np.ndarray,  # camera-radar extrinsic matrix
    h_inv_mat: np.ndarray,  # camera-radar inverse extrinsic matrix
    pixels: np.ndarray,
    radar_range: float,
    radar_azimuth: float,  # unused
) -> np.ndarray:
    js = camera_matrix_inv @ np.vstack((pixels, np.ones((1, pixels.shape[1]))))

    points = []
    for j in js.T:
        pol = [
            np.sum(j**2),
            -2 * np.sum(j * h_mat[0:3, -1]),
            np.sum(h_mat[0:3, -1] ** 2) - radar_range**2,
        ]

        w = np.roots(pol)
        m_1 = w[0] * j
        m_2 = w[1] * j

        q_1 = (h_inv_mat @ np.append(m_1, [1]))[0:3]
        q_2 = (h_inv_mat @ np.append(m_2, [1]))[0:3]

        try:
            _, azimuth_1 = cartesian_to_range_azimuth(q_1)
            _, azimuth_2 = cartesian_to_range_azimuth(q_2)

            if q_1[0] > 0 and q_2[0] < 0:
                points.append(q_1)
            elif q_1[0] < 0 and q_2[0] > 0:
                points.append(q_2)
            elif abs(radar_azimuth - azimuth_1) < abs(radar_azimuth - azimuth_2):
                points.append(q_1)
            elif abs(radar_azimuth - azimuth_2) < abs(radar_azimuth - azimuth_1):
                points.append(q_2)
        except TypeError:
            points.append(q_1 * np.nan)

    return np.array(points)


def calculate_mean_error(
    array_1: np.ndarray,
    array_2: np.ndarray,
) -> tuple[float, float]:
    # Ensure the input arrays have the correct shape
    assert array_1.shape == array_2.shape, "Both arrays must have shape (n, 3)"

    distance = np.linalg.norm(array_1 - array_2, axis=1)
    mean = np.nanmean(distance)
    std_dev = np.nanstd(distance)
    return float(mean), float(std_dev)


def main() -> None:
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = resolve_dataset_path(args)

    calibration_setup, x_result = run_calibration(
        calibration_path,
        HD_1080,
        X0,
        range_method=RangeMethod.CAMERA,
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
            ),
        )

    ground_truth = np.array(ground_truth)
    reprojected = np.array(reprojected).squeeze(1)
    error_3d = calculate_mean_error(ground_truth, reprojected)
    error_2d = calculate_mean_error(ground_truth[:, 0:2], reprojected[:, 0:2])
    print(f"3D error: {error_3d[0]:.3f} ± {error_3d[1]:.3f}")
    print(f"2D error: {error_2d[0]:.3f} ± {error_2d[1]:.3f}")


if __name__ == "__main__":
    main()
