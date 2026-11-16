"""Channel B: state and action in the nominal base frame (exact action conjugation)."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from poseshift.geometry import invert
from poseshift.waco_runtime import WacoRuntime


def runtime(dT_b, T_wb0):
    rt = WacoRuntime(cal=None)
    rt.dT_b = dT_b
    rt.C = T_wb0 @ dT_b @ invert(T_wb0)
    rt.use_base = True
    return rt


def planar(x, y, yaw_deg):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:2, 3] = (x, y)
    return T


T_WB0 = planar(-0.66, 0.0, 0.0)
T_WB0[2, 3] = 0.91


def test_base_shift_becomes_an_object_displacement_in_the_nominal_frame():
    dT_b = planar(0.06, -0.08, 12.0)
    rt = runtime(dT_b, T_WB0)
    T_wb = T_WB0 @ dT_b                                   # where the base really is
    p_true_world = np.array([0.05, 0.12, 1.02])           # end effector in the world
    # the moved robot reports its pose in the frame it believes it is in (base-frame protocol)
    p_believed = (T_WB0 @ invert(T_wb) @ np.append(p_true_world, 1.0))[:3]
    assert np.allclose(rt.canon_pos(p_believed), p_true_world)


def test_action_conjugation_is_exact():
    dT_b = planar(0.03, 0.02, 15.0)
    rt = runtime(dT_b, T_WB0)
    a = np.array([0.4, -0.2, 0.1, 0.3, -0.1, 0.25, 1.0])  # delta in the nominal frame
    out = rt.decanon_action(a)
    Rd = rt.C[:3, :3]
    assert np.allclose(out[:3], Rd.T @ a[:3])
    # R Exp(w) R^T = Exp(R w): rotating the rotation vector is the exact change of frame
    lhs = Rd.T @ R.from_rotvec(a[3:6]).as_matrix() @ Rd
    assert np.allclose(lhs, R.from_rotvec(out[3:6]).as_matrix(), atol=1e-12)
    assert out[6] == a[6]


def test_quaternion_stays_in_the_input_hemisphere():
    rt = runtime(planar(0.0, 0.0, 8.0), T_WB0)
    q = R.from_euler("xyz", [179.0, 1.0, 20.0], degrees=True).as_quat()   # the end effector points down
    for sign in (1.0, -1.0):
        out = rt.canon_quat_xyzw(sign * q)
        assert np.dot(out, sign * q) >= 0


def test_disabled_channel_is_the_identity():
    rt = WacoRuntime(cal=None)
    p, a = np.array([0.1, 0.2, 0.3]), np.arange(7.0)
    assert np.array_equal(rt.canon_pos(p), p)
    assert np.array_equal(rt.decanon_action(a), a)
