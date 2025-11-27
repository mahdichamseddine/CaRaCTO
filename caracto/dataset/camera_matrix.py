import cv2
import numpy as np


def get_camera_matrix(
    camera_matrix: np.ndarray,
    dist_coeff: np.ndarray,
    image_dimensions: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height = image_dimensions[0]
    width = image_dimensions[1]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeff, (width, height), 1, (width, height)
    )
    x, y, w, h = roi

    return (
        new_camera_matrix,
        np.linalg.inv(new_camera_matrix),  # type: ignore
        (x, y, w, h),
    )


def undistort_and_crop(
    input_image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeff: np.ndarray,
    new_camera_matrix: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    x, y, w, h = roi
    output_image = cv2.undistort(
        input_image,
        camera_matrix,
        dist_coeff,
        None,
        new_camera_matrix,
    )
    return output_image[y : y + h, x : x + w]
