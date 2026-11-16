"""Two-anchor fusion: T_bc = dT_b^-1 T_bc0 dT_c, and the gauge ambiguity of the arm anchor alone."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from poseshift.estimator import fuse, pose_error, se2_project
from poseshift.geometry import invert


def se3(t, rpy_deg):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


T_BC0 = se3([0.9, 0.05, 0.75], [-140.0, 3.0, 88.0])     # nominal camera-to-base extrinsic
DT_C = se3([0.04, -0.07, 0.03], [6.0, -9.0, 4.0])       # camera shift
DT_B = se3([0.07, -0.05, 0.0], [0.0, 0.0, 12.0])        # planar base shift


def test_fusion_recovers_the_base_shift_in_closed_form():
    T_bc = invert(DT_B) @ T_BC0 @ DT_C                  # what the arm anchor measures
    planar, raw, off = fuse(T_bc, DT_C, T_BC0)          # the world anchor provides dT_c
    assert np.allclose(planar, DT_B, atol=1e-9)
    assert np.allclose(raw, DT_B, atol=1e-9)
    assert off < 1e-6                                   # no out-of-plane residual: consistent


def test_gauge_ambiguity_same_extrinsic_different_shifts():
    T_bc = invert(DT_B) @ T_BC0 @ DT_C
    base_gauge_dT_c = invert(T_BC0) @ T_bc              # everything attributed to the camera
    camera_gauge_dT_b = T_BC0 @ invert(T_bc)            # everything attributed to the base
    assert np.allclose(T_BC0 @ base_gauge_dT_c, T_bc)
    assert np.allclose(invert(camera_gauge_dT_b) @ T_BC0, T_bc)
    # both explain the arm exactly, yet neither is the true pair of shifts
    assert pose_error(base_gauge_dT_c, DT_C)[0] > 0.01
    assert pose_error(camera_gauge_dT_b, DT_B)[0] > 0.01


def test_a_wrong_camera_shift_leaves_an_out_of_plane_residual():
    T_bc = invert(DT_B) @ T_BC0 @ DT_C
    wrong = DT_C @ se3([0.0, 0.0, 0.0], [0.0, 6.0, 0.0])
    _, raw, _ = fuse(T_bc, wrong, T_BC0)
    rpy = R.from_matrix(raw[:3, :3]).as_euler("xyz", degrees=True)
    assert abs(raw[2, 3]) > 0.02 or max(abs(rpy[0]), abs(rpy[1])) > 2.0   # the gate's thresholds


def test_se2_projection_keeps_translation_and_yaw():
    T = se3([0.03, -0.02, 0.01], [1.0, -2.0, 9.0])
    P = se2_project(T)
    assert np.allclose(P[:2, 3], T[:2, 3]) and P[2, 3] == 0.0
    assert np.allclose(R.from_matrix(P[:3, :3]).as_euler("xyz", degrees=True)[:2], 0.0, atol=1e-9)
