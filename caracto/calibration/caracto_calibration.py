"""CaRaCTO calibration: triple-constraint (sphere/plane/z) residual optimization."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from caracto.calibration.calibration_setup import CalibrationSetup, RangeMethod
from caracto.calibration.residuals import (
    compute_plane_residual,
    compute_sphere_residual,
    compute_z_residual,
)
from caracto.calibration.transformation import transformation_matrix
from caracto.cli import get_main_parser, resolve_dataset_path
from caracto.common import HD_1080, X0


class CaRaCTOSetup(CalibrationSetup):
    """Calibration via sphere/plane/z residuals over back-projected radar points."""

    def __init__(
        self,
        calibration_path: Path,
        image_dimensions: tuple[int, int],
        x0: list[float] | npt.NDArray[np.float64],
        range_method: RangeMethod,
        simulation_std: tuple[float, float, float] | None = None,  # r, theta, px
        subset: int | None = None,
    ) -> None:
        """Load the dataset and record which range_method feeds the target scale."""
        super().__init__(calibration_path, image_dimensions, x0, simulation_std, subset)
        self.range_method = range_method

    def compute_residuals(self, x: np.ndarray) -> np.ndarray:
        """Stack sphere/plane/z residuals for every measurement key."""
        # check if there are any frozen parameters and add them to the parameter list
        # if there are no frozen parameters x and calibration_params are the same
        calibration_params = self._get_calibration_params(x)
        _, h_inv = transformation_matrix(calibration_params)

        residuals = []
        for key in self.measurement_keys:
            camera_range, camera_pixel = self.get_camera_measurements(key)
            radar_range, radar_azimuth = self.get_radar_measurements(key)

            if self.range_method == RangeMethod.RADAR:
                w = self.__calculate_target_scale(camera_pixel, radar_range)
            elif self.range_method == RangeMethod.CAMERA:
                w = self.__calculate_target_scale(camera_pixel, camera_range)
            else:
                msg = "Undefined value"
                raise RuntimeError(msg)

            radar_3d, _ = self.calculate_3d_radar(h_inv, camera_pixel, w)
            res_s = compute_sphere_residual(radar_3d, radar_range)
            res_p = compute_plane_residual(radar_3d, radar_azimuth)
            res_z = compute_z_residual(radar_3d)  # Added residual

            residuals.append(res_s)
            residuals.append(res_p)
            residuals.append(res_z)

        return np.array(residuals)

    def __calculate_target_scale(
        self,
        pixel_coords: np.ndarray,
        target_dist: float,
    ) -> float:
        _, cos_angle = self._calculate_pixel_angles(pixel_coords)
        return cos_angle * target_dist


def main() -> None:
    """Run CaRaCTO calibration and print the resulting extrinsic transform."""
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = resolve_dataset_path(args)

    frozen_params = None

    calibration_setup = CaRaCTOSetup(
        calibration_path,
        HD_1080,
        X0,
        range_method=RangeMethod.CAMERA,
        subset=None,
    )
    ls_result = least_squares(
        calibration_setup.compute_residuals,
        calibration_setup.get_initial_guess(frozen_params),
        method="lm",
        verbose=0,
    )

    print(ls_result.cost)
    print(ls_result.x)
    print(ls_result.optimality)

    x_result = calibration_setup.check_frozen_params(ls_result.x)
    h_result, _h_result_inv = transformation_matrix(x_result)
    x_init = calibration_setup.check_frozen_params(np.array(X0))
    h_init, _h_init_inv = transformation_matrix(x_init)
    print(h_init)
    print(h_result)

    initial = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])
    print(np.allclose(np.round(h_result), initial))


if __name__ == "__main__":
    main()
