"""The base-frame protocol: clean scenes are untouched, a moved base reports the pose it believes."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from poseshift.frames import FrameAdapter


def fake_sim(pos, yaw_deg):
    mat = R.from_euler("z", yaw_deg, degrees=True).as_matrix()
    model = SimpleNamespace(body_name2id=lambda name: 0)
    data = SimpleNamespace(body_xpos=np.array([pos], float), body_xmat=np.array([mat.reshape(-1)], float))
    return SimpleNamespace(model=model, data=data)


NOM_POS, NOM_QUAT = np.array([-0.66, 0.0, 0.91]), np.array([1.0, 0.0, 0.0, 0.0])   # (w, x, y, z)


def test_clean_condition_matches_the_original_data():
    fa = FrameAdapter("base")
    fa.sync(fake_sim(NOM_POS, 0.0), nominal_pos=NOM_POS, nominal_quat_wxyz=NOM_QUAT)
    p = np.array([0.1, -0.2, 1.1])
    a = np.array([0.3, -0.1, 0.2, 0.05, 0.0, -0.02, 1.0])
    assert np.allclose(fa.obs_pos(p), p, rtol=0.0, atol=1e-12)
    assert np.allclose(fa.action_to_world(a), a, rtol=0.0, atol=1e-12)


def test_moved_base_reports_believed_pose_and_rotates_actions():
    yaw = 15.0
    shift = np.array([0.06, -0.08, 0.0])
    fa = FrameAdapter("base")
    fa.sync(fake_sim(NOM_POS + shift, yaw), nominal_pos=NOM_POS, nominal_quat_wxyz=NOM_QUAT)
    # a point rigidly attached to the base keeps its base-frame coordinates
    offset = np.array([0.5, 0.1, 0.2])
    p_world = NOM_POS + shift + R.from_euler("z", yaw, degrees=True).apply(offset)
    assert np.allclose(fa.obs_pos(p_world), NOM_POS + offset)
    # a base-frame action is executed in the moved base frame
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.2, -1.0])
    w = fa.action_to_world(a)
    assert np.allclose(w[:3], R.from_euler("z", yaw, degrees=True).apply(a[:3]))
    assert np.allclose(fa.action_to_policy(w), a)
    assert w[6] == a[6]


def test_world_frame_convention_ignores_the_base():
    fa = FrameAdapter("world")
    fa.sync(fake_sim(NOM_POS + [0.1, 0.0, 0.0], 15.0), nominal_pos=NOM_POS, nominal_quat_wxyz=NOM_QUAT)
    p = np.array([0.1, -0.2, 1.1])
    assert np.array_equal(fa.obs_pos(p), p)
