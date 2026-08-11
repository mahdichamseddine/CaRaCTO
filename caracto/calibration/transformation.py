"""Conversion between the 6-parameter extrinsic vector and 4x4 homogeneous transform."""

import numpy as np


def transformation_matrix(
    extrinsic_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 4x4 homogeneous transform and its inverse from extrinsic_params."""
    r_mat = rotation_matrix(extrinsic_params[0:3])
    t_vec = np.expand_dims(extrinsic_params[3:6], axis=0).T

    h_mat = np.hstack((r_mat, t_vec))
    h_mat = np.vstack((h_mat, [0, 0, 0, 1]))
    h_mat_inv = np.hstack((r_mat.T, -r_mat.T @ t_vec))
    h_mat_inv = np.vstack((h_mat_inv, [0, 0, 0, 1]))

    return h_mat, h_mat_inv


def rotation_matrix(
    euler_angles: np.ndarray,
    *,
    closed_form: bool = False,
) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles (x, y, z order).

    https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    """
    alpha = euler_angles[0]
    c_a, s_a = np.cos(alpha), np.sin(alpha)
    beta = euler_angles[1]
    c_b, s_b = np.cos(beta), np.sin(beta)
    gamma = euler_angles[2]
    c_g, s_g = np.cos(gamma), np.sin(gamma)

    if closed_form:  # Closed form rotation matrix
        r_mat = np.array([
            [c_b * c_g, s_a * s_b * c_g - c_a * s_g, c_a * s_b * c_g + s_a * s_g],
            [c_b * s_g, s_a * s_b * s_g + c_a * c_g, c_a * s_b * s_g - s_a * c_g],
            [-s_b, s_a * c_b, c_a * c_b],
        ])
    else:  # Rotation matrix as a result of matrix multiplication
        rx = np.array([[1, 0, 0], [0, c_a, -s_a], [0, s_a, c_a]])
        ry = np.array([[c_b, 0, s_b], [0, 1, 0], [-s_b, 0, c_b]])
        rz = np.array([[c_g, -s_g, 0], [s_g, c_g, 0], [0, 0, 1]])
        r_mat = rz @ ry @ rx

    return r_mat
