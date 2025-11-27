from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from caracto.calibration.calibration_setup import CalibrationSetup
from caracto.calibration.caracto_calibration import RangeMethod
from caracto.calibration.run_calibration import run_calibration
from caracto.calibration.transformation import transformation_matrix
from caracto.cli import get_main_parser
from caracto.common import HD_1080, X0
from caracto.dataset.data_loader import load_sample_data
from caracto.reconstruction.correspondences import (
    compute_correspondences,
    get_box_prompt,
    project_3d_to_2d,
)
from caracto.reconstruction.point_reconstruction import compute_point_3d
from caracto.reconstruction.spherical_cartesian import cartesian_to_range_azimuth


def get_scaling_reference(
    target_points_3d: np.ndarray,
    image_shape: np.ndarray,
    calibration_setup: CalibrationSetup,
    h_result: np.ndarray,
    measurement_key: str | None = None,
):
    if measurement_key is not None:
        # If measurement key is provided then return the ground truth data
        gt_point = np.array(calibration_setup.optitrack_data[measurement_key])
        _, gt_pixel = calibration_setup.get_camera_measurements(measurement_key)
        gt_range, _ = cartesian_to_range_azimuth(gt_point)
        return gt_pixel.squeeze(), gt_range

    target_points_mean = np.mean(target_points_3d, axis=0)
    # target_points_std = np.std(points_3d, axis=0)
    target_pixels_mean = project_3d_to_2d(
        target_points_mean[np.newaxis, :],
        image_shape,
        calibration_setup.camera_matrix,
        h_result,
    ).squeeze()
    estimated_range, _ = cartesian_to_range_azimuth(target_points_mean)

    return target_pixels_mean, estimated_range


def disparity_to_depth(
    disparity: np.ndarray,
    target_pixel: np.ndarray,
    target_range: float,
    intrinsics_matrix: np.ndarray,
) -> tuple[float, np.ndarray]:
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    # cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

    # disparity = cv2.blur(disparity, (3, 3))
    target_x, target_y = target_pixel[0], target_pixel[1]
    # f = fx
    f = (fx + fy) / 2
    b = (target_range * disparity[target_y, target_x]) / f

    scaled_depth_map = (b * f) / np.maximum(disparity, 1)

    return b, scaled_depth_map


def depth_to_3d(
    depth_map: np.ndarray,
    rgb_image: np.ndarray,
    intrinsics_matrix_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Ensure depth map and rgb image have the same dimensions
    assert depth_map.shape == rgb_image.shape[:2]

    height, width = depth_map.shape

    x = np.arange(width)
    y = np.arange(height)
    u, v = np.meshgrid(x, y)
    coord = np.stack((u, v), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1).reshape(-1, 3)

    points = (depth_map.reshape(-1) * (intrinsics_matrix_inv @ coord.T)).T

    # Get the colors

    colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    return points, colors


def compute_scene_3d(
    disparity: np.ndarray,
    rgb_image: np.ndarray,
    target_pixel: np.ndarray,
    target_range: float,
    intrinsics_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, depth_map = disparity_to_depth(
        disparity,
        target_pixel,
        target_range,
        intrinsics_matrix,
    )

    scene_points, scene_colors = depth_to_3d(
        depth_map,
        rgb_image,
        np.linalg.inv(intrinsics_matrix),
    )

    return scene_points, scene_colors


def find_planes(
    input_pcd: o3d.geometry.PointCloud,
    max_plane_idx: int,
    pt_to_plane_dist: float = 0.1,
    return_pcd: bool = False,
):
    plane_models = {}
    plane_pcds = {}
    rest_pcd = input_pcd  # .voxel_down_sample(voxel_size=0.05)
    for i in range(max_plane_idx):
        plane_models[i], inliers = rest_pcd.segment_plane(
            distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000
        )
        if return_pcd:
            colors = plt.get_cmap("tab20")(i)
            plane_pcds[i] = rest_pcd.select_by_index(inliers)
            plane_pcds[i].paint_uniform_color(list(colors[:3]))
        rest_pcd = rest_pcd.select_by_index(inliers, invert=True)

    return plane_models, plane_pcds


def optimize_scene_disparity(
    unscaled_disparity: np.ndarray,
    rgb_image: np.ndarray,
    target_pixel: np.ndarray,
    target_range: float,
    intrinsics_matrix: np.ndarray,
    max_shift: float = 250,
    path: Path | None = None,
) -> np.ndarray:
    def compute_manhattan_loss(disparity_shift: float):
        scene_points, _ = compute_scene_3d(
            unscaled_disparity + disparity_shift,
            rgb_image,
            target_pixel,
            target_range,
            intrinsics_matrix,
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_points)
        plane_models, _ = find_planes(pcd, max_plane_idx=4)
        cos_45_deg = (2**0.5) / 2
        loss = 0
        for i, plane_i in enumerate(plane_models.values()):
            for j, plane_j in enumerate(plane_models.values()):
                if i == j:
                    continue
                dot_product = np.dot(plane_i[0:3], plane_j[0:3])
                if np.abs(dot_product) < cos_45_deg:
                    loss += np.abs(dot_product)
                elif dot_product < 0:
                    loss += np.linalg.norm(plane_i[0:3] + plane_j[0:3])
                else:
                    loss += np.linalg.norm(plane_i[0:3] - plane_j[0:3])

        return loss / 2  # divide by 2 due to the double for-loops

    if (path is None) or (not path.exists()):
        rounds = []
        for i in range(1):  # TODO more runs for smoother output?
            losses = []
            for j in range(0, int(max_shift + 1)):
                # print(i, j)
                losses.append(compute_manhattan_loss(j))
            rounds.append(losses)
        rounds = np.array(rounds)

        if path is not None:
            np.save(path, rounds)
    else:
        rounds = np.load(path)

    # TODO smooth?
    optimized_shift = np.argmin(np.mean(rounds, axis=0))

    return unscaled_disparity + optimized_shift


def main() -> None:
    parser = get_main_parser()
    args = parser.parse_args()
    calibration_path = args.dataset_path

    calibration_setup, x_result = run_calibration(
        calibration_path, HD_1080, X0, range_method=RangeMethod.CAMERA
    )
    h_result, h_result_inv = transformation_matrix(x_result)
    # fmt: off
    # The following measurements achieved "good" correspondences
    # Used for faster debugging
    valid_measurements = ["Meas_03", "Meas_04", "Meas_05", "Meas_07", "Meas_09",
                          "Meas_10", "Meas_11", "Meas_14", "Meas_15", "Meas_18",
                          "Meas_19", "Meas_20", "Meas_21", "Meas_22", "Meas_23",
                          "Meas_24", "Meas_25", "Meas_26", "Meas_27", "Meas_28",
                          "Meas_33", "Meas_34", "Meas_35", "Meas_36", "Meas_37",
                          "Meas_38", "Meas_39","Meas_40"]
    # fmt: on

    for key in calibration_setup.measurement_keys:
        if key not in valid_measurements:
            continue

        radar_range, radar_azimuth, input_image = load_sample_data(
            calibration_path, calibration_setup, key
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

        unscaled_disparity, target_pixels, _ = compute_correspondences(
            input_image, area_prompt, debug=False
        )

        target_points_3d = compute_point_3d(
            calibration_setup.camera_matrix_inv,
            h_result,
            h_result_inv,
            target_pixels,
            radar_range,
            radar_azimuth,
        )

        target_pixels_mean, estimated_range = get_scaling_reference(
            target_points_3d,
            np.array(input_image.shape),
            calibration_setup,
            h_result,
            # key,  # Used for debugging
        )

        shifted_disparity = optimize_scene_disparity(
            unscaled_disparity,
            input_image,
            target_pixels_mean,
            estimated_range,
            calibration_setup.camera_matrix,
            max_shift=350,
            path=Path(calibration_path / "RadarData" / (key + ".npy")),
        )

        scene_points, scene_colors = compute_scene_3d(
            shifted_disparity,
            input_image,
            target_pixels_mean,
            estimated_range,
            calibration_setup.camera_matrix,
        )

        # Transform into radar coordiante system
        scene_points = (
            np.hstack((scene_points, np.ones((scene_points.shape[0], 1)))) @ h_result
        )[:, 0:3]

        # Original point cloud
        scene_colors = scene_colors[scene_points[:, 0] < 50, :]
        scene_points = scene_points[scene_points[:, 0] < 50, :]
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
        scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors / 255)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.25, origin=[0, 0, 0]
        )
        # Visualize original point cloud
        # o3d.visualization.draw_geometries([scene_pcd, mesh])  # type: ignore

        # Filter and align normals
        _, ind = scene_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=5)
        scene_pcd = scene_pcd.select_by_index(ind)
        scene_pcd.estimate_normals()
        scene_pcd.orient_normals_to_align_with_direction()
        # Visualize modified point cloud
        o3d.visualization.draw_geometries([scene_pcd, mesh])  # type: ignore

        # Calculate room statistics
        room_width = np.max(np.asarray(scene_pcd.points)[:, 1]) - np.min(
            np.asarray(scene_pcd.points)[:, 1]
        )
        room_height = np.max(np.asarray(scene_pcd.points)[:, 2]) - np.min(
            np.asarray(scene_pcd.points)[:, 2]
        )
        print(f"Estimated room width: {room_width:.2f} m")
        print(f"Estimated room height: {room_height:.2f} m")

        # Find and visualize planes
        _, plane_pcds = find_planes(scene_pcd, 4, 0.025, True)
        o3d.visualization.draw_geometries([plane_pcds[i] for i in range(4)])  # type: ignore


if __name__ == "__main__":
    main()
