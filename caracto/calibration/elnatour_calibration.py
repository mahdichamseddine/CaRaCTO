from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, squareform

from caracto.calibration.calibration_setup import CalibrationSetup
from caracto.calibration.residuals import (
    compute_plane_residual,
    compute_sphere_residual,
)
from caracto.calibration.transformation import transformation_matrix
from caracto.cli import get_main_parser, resolve_dataset_path
from caracto.common import HD_1080, X0

CITATION = """
@article{el2015toward,
    title     = {Toward 3D reconstruction of outdoor scenes using an MMW radar and
                  a monocular vision sensor},
    author    = {El Natour, Ghina and Ait-Aider, Omar and Rouveure, Raphael and
                  Berry, François and Faure, Patrice},
    journal   = {Sensors},
    volume    = {15},
    number    = {10},
    pages     = {25937--25967},
    year      = {2015},
    publisher = {MDPI},
}
"""


class ElNatourSetup(CalibrationSetup):
    def __init__(
        self,
        calibration_path: Path,
        image_dimensions: tuple[int, int],
        x0: list[float] | npt.NDArray[np.float64],
        simulation_std: tuple[float, float, float] | None = None,  # r, theta, px
        subset: int | None = None,
    ) -> None:
        super().__init__(calibration_path, image_dimensions, x0, simulation_std, subset)

        self.distance_matrix = self.__init_distance_matrix()
        self.cosines_matrix = self.__init_cosines_matrix()
        self.w = self.__init_scale_array()

    def compute_residuals(self, x: np.ndarray) -> np.ndarray:
        # check if there are any frozen parameters and add them to the parameter list
        # if there are no frozen parameters x and calibration_params are the same
        calibration_params = self._get_calibration_params(x)
        _, h_inv = transformation_matrix(calibration_params)

        residuals = []
        for key in self.measurement_keys:
            _, camera_pixel = self.get_camera_measurements(key)
            radar_range, radar_azimuth = self.get_radar_measurements(key)

            w = self.w[self.measurement_keys.index(key)]

            radar_3d, _ = self.calculate_3d_radar(h_inv, camera_pixel, w)
            res_s = compute_sphere_residual(radar_3d, radar_range)
            res_p = compute_plane_residual(radar_3d, radar_azimuth)

            residuals.append(res_s)
            residuals.append(res_p)

        return np.array(residuals)

    def __init_distance_matrix(self) -> np.ndarray:
        optitrack_markers = [self.optitrack_data[key] for key in self.measurement_keys]
        optitrack_markers = np.array(optitrack_markers)
        return squareform(pdist(optitrack_markers))

    def __init_cosines_matrix(self) -> np.ndarray:
        target_pixels = [self.camera_data[key][2] for key in self.measurement_keys]
        cosines = np.zeros((len(self.measurement_keys), len(self.measurement_keys)))
        for i in range(len(self.measurement_keys)):
            for j in range(len(self.measurement_keys)):
                if not target_pixels[i] or not target_pixels[j]:
                    cosines[i, j] = -1
                elif target_pixels[i] == target_pixels[j]:
                    cosines[i, j] = 1
                else:
                    _, cosines[i, j] = self._calculate_pixel_angles(
                        target_pixels[i],
                        target_pixels[j],
                    )

        return cosines

    def __radar_scale_array(self) -> np.ndarray:
        w = []
        for key in self.measurement_keys:
            radar_range = 0
            for i in self.radar_data[key]:
                radar_range += i["Range"]

            radar_range /= len(self.radar_data[key])

            _, cos_angle = self._calculate_pixel_angles(self.camera_data[key][2])
            w.append(cos_angle * radar_range)

        return np.array(w)

    def __al_kashi_method(self, w: np.ndarray) -> np.ndarray:
        residuals = []
        for i in range(len(self.measurement_keys)):
            for j in range(i + 1, len(self.measurement_keys)):
                di = w[i] / self.__cosine_wrapper(self.measurement_keys[i])
                dj = w[j] / self.__cosine_wrapper(self.measurement_keys[j])
                lij = di * dj * self.cosines_matrix[i, j]
                dij = self.distance_matrix[i, j]

                residuals.append((di**2) + (dj**2) - (2 * lij) - (dij**2))

        return np.array(residuals)

    def __init_scale_array(self) -> np.ndarray:
        w0 = self.__radar_scale_array()
        res = least_squares(self.__al_kashi_method, w0, method="lm", verbose=0)
        return res.x

    def __cosine_wrapper(self, meas_key: str) -> float:
        target_pixel = self.camera_data[meas_key][2]
        _, cosine = self._calculate_pixel_angles(target_pixel)
        return cosine


def main() -> None:
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = resolve_dataset_path(args)

    frozen_params = None

    calibration_setup = ElNatourSetup(calibration_path, HD_1080, X0, subset=None)
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
    print(np.allclose(np.round(h_init), initial))
    print(np.allclose(np.round(h_result), initial))


if __name__ == "__main__":
    main()
