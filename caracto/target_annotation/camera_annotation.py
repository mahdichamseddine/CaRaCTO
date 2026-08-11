import sys
from pathlib import Path

import cv2
import numpy as np

from caracto.cli import get_main_parser, resolve_dataset_path
from caracto.common import HD_1080
from caracto.dataset.camera_matrix import get_camera_matrix, undistort_and_crop
from caracto.dataset.file_readers import read_file
from caracto.target_annotation.drawing import draw_annotation, draw_reprojection
from caracto.target_annotation.image_enhancement import enhance_image
from caracto.target_annotation.screen_height import min_screen_height
from caracto.target_annotation.select_annotation_patch import select_patch
from caracto.target_annotation.select_annotations import select_corners


class CornerReflectorAnnotator:
    def __init__(
        self,
        camera_intrinsics: dict,
        image_dimensions: tuple[int, int] = (1080, 1920),
        corner_edges_m: tuple[float, float] = (0.077, 0.20),  # Length of corner edges
    ) -> None:
        self.screen_height = min_screen_height()
        self.corner_edges_m = corner_edges_m

        self.old_camera_matrix = np.array(camera_intrinsics["camera_matrix"])
        self.dist_coeff = np.array(camera_intrinsics["dist_coeff"])
        self.camera_matrix, _, self.roi = get_camera_matrix(
            self.old_camera_matrix,
            self.dist_coeff,
            image_dimensions,
        )

        self.annotations = {}

    def annotate_image(
        self,
        measurement_name: str,
        input_image: np.ndarray,
    ) -> tuple[dict, np.ndarray]:
        image = undistort_and_crop(
            input_image,
            self.old_camera_matrix,
            self.dist_coeff,
            self.camera_matrix,
            self.roi,
        )
        patch_corners = select_patch(image)
        if patch_corners is not None:
            image = enhance_image(image, patch_corners)
            annotation = select_corners(image, patch_corners, self.screen_height)
            annotation, reprojected = self.annotation_3d_estimation(annotation)
        else:
            annotation = {}
            reprojected = np.empty([])

        self.annotations[measurement_name] = annotation
        return annotation, reprojected

    def annotation_3d_estimation(self, annotation: dict) -> tuple[dict, np.ndarray]:
        image_points_2d, target_points_3d = self.__get_correspondences(annotation)
        _success, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=target_points_3d,
            imagePoints=image_points_2d,
            cameraMatrix=self.camera_matrix,
            distCoeffs=np.zeros((5, 1)),
            flags=0,
        )
        reprojection_3d2d, _jacobian = cv2.projectPoints(
            objectPoints=target_points_3d,
            rvec=rotation_vector,
            tvec=translation_vector,
            cameraMatrix=self.camera_matrix,
            distCoeffs=np.zeros((5, 1)),
        )

        annotation["distance_m"] = np.linalg.norm(translation_vector)

        return annotation, reprojection_3d2d

    def __get_correspondences(self, annotation: dict) -> tuple[np.ndarray, np.ndarray]:
        outer_triangle = annotation["outer_triangle"]
        inner_triangle = annotation["inner_triangle"]
        target_center = annotation["target_center"]

        image_points_2d = []
        target_points_3d = []

        image_points_2d.append(target_center[0])
        target_points_3d.append([0, 0, 0])
        # X
        image_points_2d.append(inner_triangle[0])
        target_points_3d.append([self.corner_edges_m[0], 0, 0])
        image_points_2d.append(outer_triangle[0])
        target_points_3d.append([self.corner_edges_m[1], 0, 0])
        # Z
        image_points_2d.append(inner_triangle[1])
        target_points_3d.append([0, 0, self.corner_edges_m[0]])
        image_points_2d.append(outer_triangle[1])
        target_points_3d.append([0, 0, self.corner_edges_m[1]])
        # Y
        image_points_2d.append(inner_triangle[2])
        target_points_3d.append([0, self.corner_edges_m[0], 0])
        image_points_2d.append(outer_triangle[2])
        target_points_3d.append([0, self.corner_edges_m[1], 0])

        return np.array(image_points_2d, dtype=float), np.array(
            target_points_3d,
            dtype=float,
        )

    def save_annotations(self, path: Path) -> None:
        # TODO
        pass


def main() -> None:
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = resolve_dataset_path(args)

    radar_path = calibration_path / "RadarData"
    intrinsics_path = calibration_path / "calibration_matrix.yaml"
    optitrack_path = calibration_path / "optitrack_data_transformed.json"

    camera_intrinsics = read_file(intrinsics_path)
    annotator = CornerReflectorAnnotator(camera_intrinsics, HD_1080)

    optitrack_data = read_file(optitrack_path)

    for key, item in optitrack_data.items():
        if key == "origin":
            continue

        print(key)
        optitrack_marker = np.array(item)
        print(np.linalg.norm(optitrack_marker))

        radar_file = radar_path / (key + ".pickle")
        output_test = read_file(radar_file)

        for raw_image in output_test["Camera"]:
            annotation, reprojection_3d2d = annotator.annotate_image(key, raw_image)
            image = undistort_and_crop(
                raw_image,
                annotator.old_camera_matrix,
                annotator.dist_coeff,
                annotator.camera_matrix,
                annotator.roi,
            )
            if len(annotation):
                print(annotation["distance_m"])
                image_anns = draw_annotation(image, annotation, annotate_mode=False)
                draw_reprojection(image, reprojection_3d2d)
                cv2.imshow("Video", image_anns)
            else:
                cv2.imshow("Video", image)

            key_press = cv2.waitKey() & 0xFF
            if key_press == ord("c"):  # cancel
                cv2.destroyWindow("Video")
                break
            if key_press == ord("q"):
                cv2.destroyWindow("Video")
                sys.exit()


if __name__ == "__main__":
    main()
