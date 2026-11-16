"""Error-aware fine-tuning: structured state noise leaves actions and the gripper untouched."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from poseshift.rt_aug import RTAugmenter, RTConfig, closest_rotvec


def test_default_probabilities_and_error_model():
    cfg = RTConfig()
    assert cfg.p_rt == 0.5 and cfg.p_state == 0.5
    assert cfg.cam_t == 0.12 and cfg.cam_r == 12.0                  # virtual camera: 12 cm, 12 deg per axis
    assert cfg.err_t == 0.008 and cfg.err_r == 0.6                  # estimator error model: 8 mm, 0.6 deg
    assert cfg.base_err_t == 0.008 and cfg.base_err_r == 0.6


def test_closest_rotvec_is_continuous_near_pi():
    ref = np.array([3.10, 0.02, -0.01])                             # end effector pointing down
    r = (R.from_euler("z", 0.5, degrees=True) * R.from_rotvec(ref)).as_rotvec()
    flipped = r * (1.0 - 2.0 * np.pi / np.linalg.norm(r))           # the other representation
    for cand in (r, flipped):
        out = closest_rotvec(cand, ref)
        assert np.linalg.norm(out - ref) < 0.1


def test_state_noise_is_a_planar_base_error(tmp_path):
    torch = pytest.importorskip("torch")
    aug = RTAugmenter(RTConfig(calib_dir=str(tmp_path)), {})
    T0 = np.eye(4)
    T0[:3, 3] = (-0.66, 0.0, 0.91)
    s = torch.tensor([0.05, 0.10, 1.00, 3.10, 0.02, -0.01, 0.03, -0.03])
    torch.manual_seed(0)
    out = aug._noisy_state(s, {"T_wb0": T0})
    d = (out - s).numpy()
    assert np.allclose(d[6:], 0.0)                                   # gripper untouched
    assert abs(d[2]) < 1e-6                                          # a planar error does not change the height
    assert 0.0 < np.linalg.norm(d[:2]) < 0.08                        # millimetres to a few centimetres
    assert np.linalg.norm(d[3:6]) < 0.1                              # no axis-angle flip
