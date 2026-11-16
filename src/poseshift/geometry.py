"""SE(3) helpers with explicit quaternion conventions.

MuJoCo stores quaternions as (w, x, y, z); SciPy uses (x, y, z, w). Every conversion in
this project goes through these two functions so the ordering is never guessed.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = [
    "mj2scipy",
    "scipy2mj",
    "rotmat_from_mjquat",
    "mjquat_from_rotmat",
    "pose_to_matrix",
    "matrix_to_pose",
    "invert",
    "camera_extrinsics",
    "project_point",
]


def mj2scipy(q_wxyz) -> R:
    """MuJoCo (w, x, y, z) -> scipy Rotation."""
    q = np.asarray(q_wxyz, dtype=float)
    return R.from_quat([q[1], q[2], q[3], q[0]])


def scipy2mj(rot: R) -> np.ndarray:
    """scipy Rotation -> MuJoCo (w, x, y, z)."""
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z], dtype=float)


def rotmat_from_mjquat(q_wxyz) -> np.ndarray:
    return mj2scipy(q_wxyz).as_matrix()


def mjquat_from_rotmat(mat) -> np.ndarray:
    return scipy2mj(R.from_matrix(np.asarray(mat, dtype=float)))


def pose_to_matrix(pos, rot) -> np.ndarray:
    """(3,) position + Rotation/3x3/quat -> 4x4 homogeneous matrix."""
    T = np.eye(4)
    if isinstance(rot, R):
        T[:3, :3] = rot.as_matrix()
    else:
        rot = np.asarray(rot, dtype=float)
        T[:3, :3] = rot if rot.shape == (3, 3) else rotmat_from_mjquat(rot)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def matrix_to_pose(T) -> tuple[np.ndarray, R]:
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), R.from_matrix(T[:3, :3])


def invert(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def camera_extrinsics(sim, camera_name: str) -> np.ndarray:
    """4x4 world->camera-body transform for a MuJoCo camera (camera looks down -z)."""
    cid = sim.model.camera_name2id(camera_name)
    pos = sim.data.cam_xpos[cid].copy()
    mat = sim.data.cam_xmat[cid].reshape(3, 3).copy()
    return invert(pose_to_matrix(pos, mat))


def project_point(sim, camera_name: str, p_world, width: int, height: int):
    """Pinhole projection of a world point into image pixels, or None if behind the camera.

    Returns pixel coordinates in the RAW MuJoCo image (origin at the bottom-left, y up).
    """
    cid = sim.model.camera_name2id(camera_name)
    cam_pos = sim.data.cam_xpos[cid]
    cam_mat = sim.data.cam_xmat[cid].reshape(3, 3)
    p_cam = cam_mat.T @ (np.asarray(p_world, dtype=float) - cam_pos)
    if p_cam[2] >= -1e-9:
        return None
    f = 0.5 * height / np.tan(np.deg2rad(sim.model.cam_fovy[cid]) / 2.0)
    u = width / 2.0 + f * p_cam[0] / -p_cam[2]
    v = height / 2.0 - f * p_cam[1] / -p_cam[2]
    return np.array([u, v])
