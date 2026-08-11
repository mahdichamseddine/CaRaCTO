import random
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from caracto.dataset.camera_matrix import get_camera_matrix
from caracto.dataset.caracto_dataset import CaractoDataset
from caracto.reconstruction.spherical_cartesian import cartesian_to_range_azimuth

_rng = np.random.default_rng()

NUM_CALIBRATION_PARAMS = 6  # 3 rotation + 3 translation extrinsic parameters


class RangeMethod(Enum):
    RADAR = 0
    CAMERA = 1


class CalibrationSetup:
    def __init__(
        self,
        calibration_path: Path,
        image_dimensions: tuple[int, int],
        x0: list[float] | npt.NDArray[np.float64],
        simulation_std: tuple[float, float, float] | None = None,  # r, theta, px
        subset: int | None = None,
    ) -> None:
        if isinstance(x0, list):
            assert len(x0) == NUM_CALIBRATION_PARAMS, "Expected 6 initial values"
            x0 = np.array(x0)
        elif isinstance(x0, np.ndarray):
            assert x0.shape[0] == NUM_CALIBRATION_PARAMS, "Expected 6 initial values"

        self.x0 = x0
        self.frozen_params = None

        self.dataset = CaractoDataset(calibration_path)

        camera_intrinsics = self.dataset.load_camera_intrinsics()
        self.old_camera_matrix = np.array(camera_intrinsics["camera_matrix"])
        self.dist_coeff = np.array(camera_intrinsics["dist_coeff"])
        self.camera_matrix, self.camera_matrix_inv, self.roi = get_camera_matrix(
            self.old_camera_matrix,
            self.dist_coeff,
            image_dimensions,
        )

        # Rebuilt in their pre-migration shapes (list-of-lists / 3-vector / 4-tuple
        # with a dict-list) so the rest of this class, and callers like
        # elnatour_calibration.py, run_evaluation.py, and point_reconstruction.py,
        # don't need to change at all.
        self.camera_data = {
            key: self.__legacy_camera_data_entry(key)
            for key in self.dataset.keys()  # noqa: SIM118 (CaractoDataset, not a dict)
        }
        self.optitrack_data = {
            key: self.dataset.load_ground_truth(key)["target_center_xyz"]
            for key in self.dataset.keys()  # noqa: SIM118 (CaractoDataset, not a dict)
        }
        self.radar_data = {
            key: self.__radar_detection_dicts(key)
            for key in self.dataset.keys()  # noqa: SIM118 (CaractoDataset, not a dict)
        }
        self.valid_for_reconstruction_keys = {
            key
            for key in self.dataset.keys()  # noqa: SIM118 (CaractoDataset, not a dict)
            if self.dataset.positions[key]["valid_for_reconstruction"]
        }

        self.measurement_keys = self.__get_measurement_keys(subset)

        # Standard deviatio to the normal distribution for adding simulation noise
        self.simulation_std = simulation_std
        self.__simulated_radar = {}
        self.__simulated_camera = {}

    def __legacy_camera_data_entry(self, key: str) -> list:
        annotation = self.dataset.load_annotation(key)
        if annotation is None:
            return []
        return [
            annotation["outer_triangle"],
            annotation["inner_triangle"],
            [annotation["target_center"]],
            annotation["corner_edges"],
            annotation["distance_m"],
        ]

    def __radar_detection_dicts(self, key: str) -> list[dict]:
        detections = self.dataset.load_radar_detections(key)
        amplitude = (detections.amplitude_re + 1j * detections.amplitude_im).astype(
            np.complex64,
        )
        return [
            {
                "Range": detections.range_m[i],
                "Vel": detections.velocity_mps[i],
                "Mag": detections.magnitude[i],
                "Ang": detections.angle_rad[i],
                "Noise": detections.noise[i],
                "Amp": amplitude[i],
            }
            for i in range(len(detections))
        ]

    def compute_residuals(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_initial_guess(
        self,
        fixed_params: list[bool] | npt.NDArray[np.bool_] | None = None,
    ) -> np.ndarray:
        if fixed_params is None:
            return self.x0
        if isinstance(fixed_params, list):
            fixed_params = np.array(fixed_params)

        assert fixed_params.shape == self.x0.shape, (
            "fixed_params and x0 must have the same shape"
        )

        assert fixed_params.dtype == bool, "fixed_params must be a bool array"

        self.frozen_params = fixed_params
        return self.x0[~self.frozen_params]

    def calculate_3d_radar(
        self,
        xform: np.ndarray,
        pixel_coords: np.ndarray,
        w: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        j = self.camera_matrix_inv @ np.append(pixel_coords, [1])
        wj = np.append(w * j, [1])

        return xform @ wj, j

    def get_camera_measurements(self, key: str) -> tuple[float, np.ndarray]:
        camera_range = self.camera_data[key][-1]
        camera_pixel = self.camera_data[key][2]

        if self.simulation_std is not None:
            if key in self.__simulated_camera:
                camera_range, camera_pixel = self.__simulated_camera[key]
            elif self.simulation_std[2] > 0:
                camera_pixel[0][0] += _rng.normal(0, self.simulation_std[2])
                camera_pixel[0][1] += _rng.normal(0, self.simulation_std[2])
                self.__simulated_camera[key] = (camera_range, camera_pixel)

        return float(camera_range), np.array(camera_pixel)

    def get_radar_measurements(self, key: str) -> tuple[float, float]:
        radar_range = 0
        radar_azimuth = 0

        if self.simulation_std is not None:
            if key in self.__simulated_radar:
                radar_range, radar_azimuth = self.__simulated_radar[key]
            else:
                ground_truth = np.array(self.optitrack_data[key])
                radar_range, radar_azimuth = cartesian_to_range_azimuth(ground_truth)

                if self.simulation_std[0] > 0:
                    radar_range += _rng.normal(0, self.simulation_std[0])
                if self.simulation_std[1] > 0:
                    radar_azimuth += _rng.normal(0, self.simulation_std[1])

                self.__simulated_radar[key] = (radar_range, radar_azimuth)

        else:
            for i in self.radar_data[key]:
                radar_range += float(i["Range"])
                radar_azimuth += float(i["Ang"])

            radar_range /= len(self.radar_data[key])
            radar_azimuth /= -len(self.radar_data[key])

        return radar_range, radar_azimuth

    def check_frozen_params(self, x_in: np.ndarray) -> np.ndarray:
        if self.frozen_params is None:
            return x_in

        x_out = self.x0.copy()
        x_out[~self.frozen_params] = x_in
        return x_out

    def _calculate_pixel_angles(
        self,
        point_1: np.ndarray,
        point_2: np.ndarray | None = None,
    ) -> tuple[float, float]:
        # If 1 point is given, the second point is chosen to be the focal point
        # If 2 points are given, the angle between them is calculated

        camera_center = self.camera_matrix[0:2, -1]
        focal_length = np.array([self.camera_matrix[0, 0], self.camera_matrix[1, 1]])

        if point_2 is None:
            vec_1 = np.array([0, 0, 1], dtype=float)
            vec = point_1 - camera_center
            vec_2 = np.append(vec / focal_length, [1])
        else:
            vec = point_1 - camera_center
            vec_1 = np.append(vec / focal_length, [1])
            vec = point_2 - camera_center
            vec_2 = np.append(vec / focal_length, [1])

        unit_vec_1 = vec_1 / np.linalg.norm(vec_1)
        unit_vec_2 = vec_2 / np.linalg.norm(vec_2)
        dot_product = np.dot(unit_vec_1, unit_vec_2)
        angle = np.arccos(dot_product)

        return float(angle), float(dot_product)

    def _get_calibration_params(self, x_in: np.ndarray) -> np.ndarray:
        if x_in.shape != self.x0.shape:
            assert self.frozen_params is not None, (
                "Too little input values and frozen_params is not defined"
            )
            x_out = self.check_frozen_params(x_in)
        else:
            x_out = x_in

        return x_out

    def __get_measurement_keys(self, subset: int | None = None) -> list[str]:
        measurement_keys = [
            key
            for key in self.dataset.keys()  # noqa: SIM118 (CaractoDataset, not a dict)
            if self.dataset.positions[key]["annotated"]
        ]

        if subset and (0 < subset < len(measurement_keys)):
            measurement_keys = random.sample(measurement_keys, subset)

        return measurement_keys
