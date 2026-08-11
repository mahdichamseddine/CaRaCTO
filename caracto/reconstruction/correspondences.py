"""Finds camera/radar target correspondences via depth estimation + segmentation."""

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d

from caracto.calibration.caracto_calibration import RangeMethod
from caracto.calibration.run_calibration import run_calibration
from caracto.calibration.transformation import transformation_matrix
from caracto.cli import get_main_parser, resolve_dataset_path
from caracto.common import HD_1080, X0
from caracto.dataset.data_loader import load_sample_data
from caracto.ml.depth_anything import depth_estimation
from caracto.ml.segment_anything import instance_segmentation
from caracto.reconstruction.point_reconstruction import compute_point_3d
from caracto.reconstruction.spherical_cartesian import (
    cartesian_to_range_azimuth,
    spherical_to_cartesian,
)
from caracto.target_annotation.drawing import draw_rectangle


def compute_prompt_limits(
    radar_range: float,
    radar_azimuth: float,
    azimuth_limit: float,
    elevation_limit: float,
) -> np.ndarray:
    """Return the two 3D corners bounding radar_range/azimuth ± the given limits."""
    lower_limit = spherical_to_cartesian(
        radar_range,
        radar_azimuth - azimuth_limit,
        -elevation_limit,
    )
    upper_limit = spherical_to_cartesian(
        radar_range,
        radar_azimuth + azimuth_limit,
        elevation_limit,
    )
    prompt_limits = [lower_limit, upper_limit]

    return np.array(prompt_limits)


def project_3d_to_2d(
    points: np.ndarray,
    image_limits: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray | None = None,
) -> npt.NDArray[np.int64]:
    """Project 3D points into clamped, integer pixel coordinates."""
    if extrinsic_matrix is None:
        extrinsic_matrix = np.identity(4)

    h, w, _ = image_limits
    pixels = []
    for point in points:
        world_3d = np.hstack((point, [1]))
        world_3d = world_3d[np.newaxis, :].T
        camera_3d = (extrinsic_matrix @ world_3d)[0:3]
        camera_2d = intrinsic_matrix @ camera_3d
        pixel = camera_2d[0:2] / camera_2d[2]

        pixel[0] = max(0, pixel[0])
        pixel[0] = min(w - 1, pixel[0])
        pixel[1] = max(0, pixel[1])
        pixel[1] = min(h - 1, pixel[1])

        pixels.append(np.round(pixel.squeeze()))
    pixels.reverse()

    return np.array(pixels, dtype=int)


def get_box_prompt(
    radar_range: float,
    radar_azimuth: float,
    image_shape: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    azimuth_range_deg: float = 10,  # ± azimuth_range_deg / 2
    elevation_range_deg: float = 15,  # ± elevation_range_deg / 2
) -> np.ndarray:
    """Return the pixel-space box prompt around the radar's expected target position."""
    # degree to radians
    azimuth_limits_rad = (azimuth_range_deg * np.pi / 180) / 2
    elevation_limits_rad = (elevation_range_deg * np.pi / 180) / 2

    prompt_limits_3d = compute_prompt_limits(
        radar_range,
        radar_azimuth,
        azimuth_limits_rad,
        elevation_limits_rad,
    )

    return project_3d_to_2d(
        prompt_limits_3d,
        image_shape,
        intrinsic_matrix,
        extrinsic_matrix,
    )


# def get_point_prompt(prompt_limits: np.ndarray) -> np.ndarray:
#     pass


def merge_segmentation(masks_1: np.ndarray, _masks_2: np.ndarray | None) -> np.ndarray:
    """Return the target mask; currently just the first depth-segmentation mask."""
    # TODO: placeholder assuming the best mask is the first in the depth segmentation
    return masks_1[0]


def compute_correspondences(
    input_image: np.ndarray,
    area_prompt: np.ndarray,
    *,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (disparity, target_pixels, target_mean) for the target in area_prompt."""
    # Depth estimation using Depth Anything V2
    normalized_depth, unscaled_disparity = depth_estimation(
        cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
        0,
    )

    visualize("Depth - Gray", normalized_depth, color=False, debug=debug)

    colored_depth = np.array(
        [
            normalized_depth * 255,
            normalized_depth * 255,
            normalized_depth * 255,
        ],
        dtype=np.uint8,
    ).transpose((1, 2, 0))
    colored_depth = cv2.applyColorMap(colored_depth, cv2.COLORMAP_PARULA)
    colored_depth_prompt = draw_rectangle(
        colored_depth,
        area_prompt[0],
        area_prompt[1],
        (255, 255, 255),
    )

    visualize("Depth - Colored", colored_depth_prompt, color=True, debug=debug)

    # Instance segmentation using Segment Anything Model
    depth_segmentation_masks = instance_segmentation(
        colored_depth,
        input_points=None,
        input_boxes=[[area_prompt.flatten().tolist()]],
    )
    for mask in depth_segmentation_masks:
        # Show the first mask only
        visualize("Depth - Masked", mask, color=False, debug=debug)
        break

    # TODO: add image segmentation using points

    target_mask = merge_segmentation(depth_segmentation_masks, None)
    x, y = np.nonzero(target_mask)
    target_pixels = np.array((y, x))
    target_mean = np.mean(target_pixels, axis=1, dtype=int).T

    return unscaled_disparity, target_pixels, target_mean


def visualize(name: str, data: np.ndarray, *, color: bool, debug: bool = True) -> None:
    """Show data in an OpenCV window named name, if debug is set."""
    if not debug:
        return

    if color:
        cv2.imshow(name, data)
    else:
        cv2.imshow(name, data.astype(np.float32))

    cv2.waitKey(0)
    cv2.destroyWindow(name)


def main() -> None:
    """Run calibration, find target correspondences per position, and print errors."""
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

    for key in calibration_setup.measurement_keys:
        if key not in calibration_setup.valid_for_reconstruction_keys:
            continue
        radar_range, radar_azimuth, input_image = load_sample_data(
            calibration_path,
            calibration_setup,
            key,
        )

        area_prompt = get_box_prompt(
            radar_range,
            radar_azimuth,
            np.array(input_image.shape),
            calibration_setup.camera_matrix,
            h_result,
            azimuth_range_deg=10,  # azimuth error range
            elevation_range_deg=15,  # elevation error range
        )

        _, target_pixels, _ = compute_correspondences(
            input_image,
            area_prompt,
            debug=True,
        )

        points_3d = compute_point_3d(
            calibration_setup.camera_matrix_inv,
            h_result,
            h_result_inv,
            target_pixels,
            radar_range,
            radar_azimuth,
        )
        points_mean = np.mean(points_3d, axis=0)
        estimated_range, estimated_azimuth = cartesian_to_range_azimuth(points_mean)
        gt_point = np.array(calibration_setup.optitrack_data[key])
        gt_range, gt_azimuth = cartesian_to_range_azimuth(gt_point)

        # Calculate errors
        print(key)
        print(
            f"range:\noptitrack: {gt_range:.2f} m\n"
            f"radar: {radar_range:.2f} m, "
            f"{(abs((gt_range - radar_range) / gt_range)) * 100:.2f}%\n"
            f"estimated: {estimated_range:.2f} m, "
            f"{(abs((gt_range - estimated_range) / gt_range)) * 100:.2f}%\n",
        )
        print(
            f"azimuth:\noptitrack: {gt_azimuth:.2f} rad\n"
            f"radar: {radar_azimuth:.2f} rad, "
            f"{(abs((gt_azimuth - radar_azimuth) / gt_azimuth)) * 100:.2f}%\n"
            f"estimated: {estimated_azimuth:.2f} rad, "
            f"{(abs((gt_azimuth - estimated_azimuth) / gt_azimuth)) * 100:.2f}%\n",
        )
        radar_error = np.linalg.norm(
            gt_point - spherical_to_cartesian(radar_range, radar_azimuth, 0),
        )
        estimated_error = np.linalg.norm(gt_point - points_mean)

        print(
            f"3D error:\nradar: {radar_error:.2f} m\n"
            f"estimated: {estimated_error:.2f} m\n",
        )

        # Visualize in 3D
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_3d)
        # target_pcd.colors
        estimated_pcd = o3d.geometry.PointCloud()
        estimated_pcd.points = o3d.utility.Vector3dVector([points_mean])
        estimated_pcd.colors = o3d.utility.Vector3dVector([[1, 1, 0]])
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector([gt_point])
        gt_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1,
            origin=[0, 0, 0],
        )

        o3d.visualization.draw_geometries(  # ty: ignore[possibly-missing-submodule]
            [target_pcd, gt_pcd, estimated_pcd, mesh],
        )


if __name__ == "__main__":
    main()
