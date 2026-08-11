import cv2
import numpy as np


def draw_annotation(
    input_image: np.ndarray,
    annotation: dict,
    *,
    annotate_mode: bool = True,
) -> np.ndarray:
    outer_triangle = annotation["outer_triangle"]
    inner_triangle = annotation["inner_triangle"]
    corner_edges = annotation["corner_edges"]
    target_center = annotation["target_center"]

    image_copy = input_image.copy()
    image_copy = draw_triangle(image_copy, outer_triangle, (255, 0, 0), 1)
    image_copy = draw_triangle(image_copy, inner_triangle, (0, 0, 255), 1)
    image_copy = draw_lines(image_copy, corner_edges, (255, 255, 255), 1)

    if annotate_mode:
        image_copy = draw_points(image_copy, outer_triangle, (255, 0, 0), 2)
        image_copy = draw_points(image_copy, inner_triangle, (0, 0, 255), 2)
        image_copy = draw_points(image_copy, target_center, (255, 255, 255), 2)

    return image_copy


def draw_lines(
    input_image: np.ndarray,
    lines: np.ndarray,
    color: tuple[int, int, int],
    line_thickness: int,
) -> np.ndarray:
    image_copy = input_image.copy()
    for line in lines:
        cv2.line(
            image_copy,
            line[0],
            line[1],
            color=color,
            thickness=line_thickness,
        )

    return image_copy


def draw_points(
    input_image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    marker_thickness: int,
) -> np.ndarray:
    image_copy = input_image.copy()
    for point in points:
        cv2.drawMarker(
            image_copy,
            (point[0], point[1]),
            color=color,
            markerType=cv2.MARKER_CROSS,
            thickness=marker_thickness,
        )

    return image_copy


def draw_rectangle(
    input_image: np.ndarray,
    start_point: tuple[int, int],
    end_point: tuple[int, int],
    color: tuple[int, int, int],
    line_thickness: int = 2,
) -> np.ndarray:
    image_copy = np.copy(input_image)
    cv2.rectangle(
        img=image_copy,
        pt1=start_point,
        pt2=end_point,
        color=color,
        thickness=line_thickness,
    )

    return image_copy


def draw_reprojection(
    input_image: np.ndarray,
    reprojected: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    marker_size: int = 5,
) -> np.ndarray:
    image_copy = input_image.copy()
    for p in reprojected.squeeze():
        cv2.drawMarker(
            image_copy,
            (int(p[0]), int(p[1])),
            color=color,
            markerType=cv2.MARKER_STAR,
            markerSize=marker_size,
        )

    return image_copy


def draw_triangle(
    input_image: np.ndarray,
    triangle: np.ndarray,
    color: tuple[int, int, int],
    line_thickness: int,
) -> np.ndarray:
    image_copy = input_image.copy()
    cv2.polylines(
        image_copy,
        [triangle],
        isClosed=True,
        color=color,
        thickness=line_thickness,
    )

    return image_copy
