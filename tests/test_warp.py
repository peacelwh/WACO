"""Channel A: depth-based reprojection between two pinhole cameras."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from poseshift.warp import focal_from_fovy, forward_splat, inverse_warp

H = W = 96
F = focal_from_fovy(45.0, H)


def scene():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 1.2)                       # fronto-parallel plane
    depth[30:60, 30:60] = 0.9                          # a box in front of it
    return rgb, depth


def shift(t, rpy_deg):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


def test_identity_warp_returns_the_image():
    rgb, depth = scene()
    for warp in (forward_splat, inverse_warp):
        out, valid = warp(rgb, depth, F, np.eye(4), np.eye(4))
        assert valid.all()
        assert np.array_equal(out, rgb)


def test_unseen_pixels_are_grey_and_never_hallucinated():
    rgb, depth = scene()
    out, valid = forward_splat(rgb, depth, F, shift([0.0, 0.0, 0.0], [0.0, 10.0, 0.0]), np.eye(4))
    assert 0.05 < (~valid).mean() < 0.5                 # a band left the original frustum
    assert np.all(out[~valid] == 128)


def test_reference_fill_uses_the_empty_workspace_view():
    rgb, depth = scene()
    ref = np.full_like(rgb, 7)
    out, seen = inverse_warp(rgb, depth, F, shift([0.0, 0.0, 0.0], [0.0, 10.0, 0.0]), np.eye(4), fill_rgb=ref)
    assert (~seen).any()
    assert np.all(out[~seen] == 7)


def test_round_trip_reproduces_the_frame_where_it_is_visible():
    rng = np.random.default_rng(1)
    rgb = np.repeat(np.repeat(rng.integers(0, 255, size=(H // 8, W // 8, 3), dtype=np.uint8), 8, 0), 8, 1)
    depth = np.full((H, W), 1.2)
    dT = shift([0.03, -0.02, 0.0], [0.0, 0.0, 0.0])
    img1, _, d1 = forward_splat(rgb, depth, F, np.eye(4), dT, return_depth=True)
    img2, valid = forward_splat(img1, d1, F, dT, np.eye(4))
    err = np.abs(img2.astype(int) - rgb.astype(int)).max(axis=2)
    assert valid.mean() > 0.9
    assert np.mean(err[valid] <= 2) > 0.8              # holes and borders aside, the content is unchanged
