from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment as munkres

from caracto.target_annotation.drawing import draw_annotation, draw_points


def select_corners(
    input_image: np.ndarray,
    patch_corners: np.ndarray,
    screen_height: int,
) -> dict:
    def select_corners_callback(
        event: int, x: int, y: int, flags: int, param: Any | None
    ) -> None:
        nonlocal p_x, p_y

        if event == cv2.EVENT_LBUTTONDOWN:
            p_x, p_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            pass

    num_corners = 6  # Two triangles containing each other
    p_x, p_y = -1, -1

    window_name = "Select Corners"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_corners_callback)

    image_patch = input_image[
        patch_corners[0, 1] : patch_corners[1, 1],
        patch_corners[0, 0] : patch_corners[1, 0],
    ]
    resize_factor = int(screen_height / (1.05 * image_patch.shape[0]))
    resized_patch = cv2.resize(image_patch, (0, 0), fx=resize_factor, fy=resize_factor)

    annotation = None
    selection_updated = False
    selected_points = []
    visualization = resized_patch
    while True:
        if p_x != -1 and p_y != -1:
            if len(selected_points) < num_corners:
                selected_points.append((p_x, p_y))
            else:
                selected_points = update_selection(selected_points, p_x, p_y)

            selection_updated = True
            p_x, p_y = -1, -1

        if selection_updated:
            if len(selected_points) < num_corners:
                visualization = draw_points(
                    resized_patch, np.array(selected_points), (255, 255, 255), 2
                )
            else:
                annotation = process_annotation(selected_points)
                visualization = draw_annotation(resized_patch, annotation)
            selection_updated = False

        cv2.imshow(window_name, visualization)
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord("c"):  # cancel
            cv2.destroyWindow(window_name)
            exit()  # TODO handle cancel properly
            # return {}
        elif key_press == ord("q"):  # exit
            cv2.destroyWindow(window_name)
            exit()
        elif key_press == ord("r"):  # restart
            p_x, p_y = -1, -1
            annotation = None
            selection_updated = False
            selected_points = []
            visualization = resized_patch
        elif key_press == ord("s"):  # save
            if annotation:
                cv2.destroyWindow(window_name)
                break

    annotation = rescale_annotation(annotation, resize_factor, patch_corners)
    return annotation


def update_selection(
    points_list: list[tuple[int, int]],
    p_x: int,
    p_y: int,
):
    # Find closest point in the lost
    new_point = np.array((p_x, p_y))
    squared_distance = np.sum((np.array(points_list) - new_point) ** 2, axis=1)
    min_idx = np.argmin(squared_distance)

    # Update the closest point in the list with the new point
    points_list.pop(min_idx)
    points_list.append((p_x, p_y))

    return points_list


def sort_points_ccw(image_points: np.ndarray) -> np.ndarray:
    # Calculate the centroid of the points
    centroid = np.mean(image_points, axis=0)

    # Compute the angle of each point with respect to the centroid, inverting the y-coordinates
    angles = np.arctan2(
        -(image_points[:, 1] - centroid[1]), image_points[:, 0] - centroid[0]
    )

    # return sorted points by angle in counterclockwise order
    return image_points[np.argsort(angles)]


def extract_triangles(
    points_list: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    min_x, min_y = np.min(points_list, axis=0)
    max_x, max_y = np.max(points_list, axis=0)

    outer_triangle = []
    inner_triangle = []
    for point in points_list:
        if len(outer_triangle) < 3 and (
            point[0] in (min_x, max_x) or point[1] in (min_y, max_y)
        ):
            outer_triangle.append(point)
        elif len(inner_triangle) < 3:
            inner_triangle.append(point)
        else:  # corner case to preven the interface from crashing
            outer_triangle.append(point)

    assert len(outer_triangle) == 3 and len(inner_triangle) == 3, (
        "Triangles must have 3 corners!"
    )

    outer_triangle = sort_points_ccw(np.array(outer_triangle))
    inner_triangle = np.array(inner_triangle)

    # Make sure the inner and outer triangle corners are aligned by index
    cost = np.sum(
        (outer_triangle[:, np.newaxis, :] - inner_triangle[np.newaxis, :, :]) ** 2,
        axis=2,
    )

    _, idx_i = munkres(cost)
    inner_triangle = inner_triangle[idx_i, :]

    return outer_triangle, inner_triangle


def calculate_center(
    outer_triangle: np.ndarray, inner_triangle: np.ndarray
) -> np.ndarray:
    intersections = []
    for i in range(3):
        for j in range(i + 1, 3):
            intersections.append(
                line_intersection(
                    outer_triangle[i],
                    inner_triangle[i],
                    outer_triangle[j],
                    inner_triangle[j],
                )
            )

    return np.mean(intersections, axis=0, dtype=int)[np.newaxis, :]


def process_annotation(marked_points: list[tuple[int, int]]) -> dict:
    outer_triangle, inner_triangle = extract_triangles(marked_points)
    target_center = calculate_center(outer_triangle, inner_triangle)
    corner_edges = get_corner_edges(outer_triangle, inner_triangle)

    annotation = {
        "outer_triangle": outer_triangle,
        "inner_triangle": inner_triangle,
        "target_center": target_center,
        "corner_edges": corner_edges,
    }

    return annotation


def get_corner_edges(outer_triangle: np.ndarray, inner_triangle: np.ndarray):
    edges = []
    for i, outer_corner in enumerate(outer_triangle):
        opposite_edge = np.delete(inner_triangle, i, axis=0)
        inner_corner = inner_triangle[i, :]
        intersection = line_intersection(
            outer_corner, inner_corner, opposite_edge[0, :], opposite_edge[1, :]
        )
        edges.append([outer_triangle[i], intersection])

    return np.array(edges)


def line_intersection(
    a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return np.array([float("inf"), float("inf")])
    return np.array([x / z, y / z]).astype(int)


def rescale_annotation(
    annotation: dict, resize_factor: int, area_corners: np.ndarray
) -> dict:
    outer_triangle = annotation["outer_triangle"]
    inner_triangle = annotation["inner_triangle"]
    corner_edges = annotation["corner_edges"]
    target_center = annotation["target_center"]

    outer_triangle = rescale_points(outer_triangle, resize_factor, area_corners)
    inner_triangle = rescale_points(inner_triangle, resize_factor, area_corners)
    target_center = rescale_points(target_center, resize_factor, area_corners)
    rescaled_edges = []
    for edge in corner_edges:
        edge = rescale_points(edge, resize_factor, area_corners)
        rescaled_edges.append(edge)

    rescaled_annotation = {
        "outer_triangle": outer_triangle,
        "inner_triangle": inner_triangle,
        "target_center": target_center,
        "corner_edges": rescaled_edges,
    }
    return rescaled_annotation


def rescale_points(
    points: np.ndarray, resize_factor, area_corners: np.ndarray
) -> np.ndarray:
    scaled_points = points / resize_factor + area_corners[0, :]
    return np.round(scaled_points).astype(int)
