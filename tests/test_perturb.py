"""The perturbation layer of LIBERO-PoseShift: deterministic sampling and level bounds."""
import numpy as np
import pytest

from poseshift.perturb import LEVELS, sample_perturbation


def test_levels_match_the_benchmark_table():
    # level -> (camera translation m, camera rotation deg / axis, base translation m, base yaw deg)
    assert LEVELS["l1"] == (0.02, 3.0, 0.02, 3.0)
    assert LEVELS["l2"] == (0.05, 5.0, 0.05, 8.0)
    assert LEVELS["l3"] == (0.10, 10.0, 0.10, 15.0)


def test_clean_is_identity():
    assert sample_perturbation("clean", "clean", "libero_spatial", 0, 0).is_identity()
    assert sample_perturbation("cam", "clean", "libero_spatial", 0, 0).is_identity()


def test_sampling_is_deterministic_and_episode_specific():
    a = sample_perturbation("both", "l3", "libero_10", 3, 7, seed=0)
    b = sample_perturbation("both", "l3", "libero_10", 3, 7, seed=0)
    c = sample_perturbation("both", "l3", "libero_10", 3, 8, seed=0)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != c.to_dict()


@pytest.mark.parametrize("level", ["l1", "l2", "l3"])
def test_shifts_stay_within_the_level(level):
    cam_t, cam_r, base_t, base_yaw = LEVELS[level]
    for ep in range(200):
        s = sample_perturbation("both", level, "libero_spatial", ep % 10, ep)
        r = np.linalg.norm(s.cam_dpos)
        assert 0.5 * cam_t - 1e-9 <= r <= cam_t + 1e-9          # between half and the full radius
        assert np.all(np.abs(s.cam_drot_rpy) <= cam_r + 1e-9)   # uniform per axis
        r_b = np.linalg.norm(s.base_dpos[:2])
        assert 0.5 * base_t - 1e-9 <= r_b <= base_t + 1e-9
        assert abs(s.base_dpos[2]) <= 0.1 * base_t + 1e-9       # the base moves in the floor plane
        assert abs(s.base_dyaw) <= base_yaw + 1e-9


def test_kinds_are_independent():
    cam = sample_perturbation("cam", "l2", "libero_spatial", 0, 0)
    base = sample_perturbation("base", "l2", "libero_spatial", 0, 0)
    assert not np.any(cam.base_dpos) and cam.base_dyaw == 0.0
    assert not np.any(base.cam_dpos) and not np.any(base.cam_drot_rpy)
