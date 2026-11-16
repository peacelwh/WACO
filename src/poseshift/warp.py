"""Depth-based forward splatting between two pinhole cameras.

All arrays use the RAW MuJoCo convention: row 0 is the BOTTOM of the image and the camera
looks down its -z axis with +x right and +y up. Unprojection and projection below use the
same convention, so it cancels out and no flips are needed; the LeRobot processor applies
its 180-degree rotation afterwards, exactly as for an unwarped frame.
"""

from __future__ import annotations

import numpy as np

__all__ = ["focal_from_fovy", "forward_splat", "inverse_warp", "real_depth"]


def focal_from_fovy(fovy_deg: float, height: int) -> float:
    return 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)


def real_depth(sim, depth_buffer: np.ndarray) -> np.ndarray:
    """Convert MuJoCo's normalised [0, 1] depth buffer into metric distance along -z."""
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    d = np.asarray(depth_buffer, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    return near / (1.0 - d * (1.0 - near / far))


def forward_splat(
    rgb: np.ndarray,
    depth: np.ndarray,
    f_px: float,
    T_world_src: np.ndarray,
    T_world_dst: np.ndarray,
    fill: int = 128,
    max_depth: float = 10.0,
    return_depth: bool = False,
    f_dst: float | None = None,
):
    """Re-render `rgb` (seen from camera `src`) as seen from camera `dst`.

    Args:
        rgb:   (H, W, 3) uint8, raw MuJoCo orientation.
        depth: (H, W) metric depth along the optical axis for the same camera.
        f_px:  focal length of the SOURCE camera in pixels (square pixels, centred principal point).
        f_dst: focal length of the destination camera (defaults to f_px).
        T_world_src / T_world_dst: 4x4 camera-to-world poses (MuJoCo camera axes).
    Returns:
        (warped rgb, valid mask[, warped depth]) -- invalid pixels are holes or outside the
        source frustum; the optional depth is along the destination camera's optical axis.
    """
    H, W = depth.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = depth.reshape(-1)
    ok = (d > 1e-4) & (d < max_depth)
    x = (cc.reshape(-1) + 0.5 - W / 2.0) * d / f_px
    y = (rr.reshape(-1) + 0.5 - H / 2.0) * d / f_px
    pts_src = np.stack([x, y, -d, np.ones_like(d)], axis=0)  # camera frame, homogeneous
    T = np.linalg.inv(T_world_dst) @ T_world_src
    p = T @ pts_src
    z = -p[2]
    ok &= z > 1e-4
    fd = f_px if f_dst is None else f_dst
    c2 = np.floor(fd * p[0] / np.maximum(z, 1e-9) + W / 2.0).astype(np.int64)
    r2 = np.floor(fd * p[1] / np.maximum(z, 1e-9) + H / 2.0).astype(np.int64)
    ok &= (c2 >= 0) & (c2 < W) & (r2 >= 0) & (r2 < H)

    idx = np.flatnonzero(ok)
    tgt = r2[idx] * W + c2[idx]
    # z-buffer: for every target pixel keep the nearest source point
    order = np.lexsort((z[idx], tgt))
    tgt_sorted = tgt[order]
    first = np.ones_like(tgt_sorted, dtype=bool)
    first[1:] = tgt_sorted[1:] != tgt_sorted[:-1]
    win_src = idx[order][first]
    win_tgt = tgt_sorted[first]

    out = np.full((H * W, 3), fill, dtype=np.uint8)
    valid = np.zeros(H * W, dtype=bool)
    out[win_tgt] = rgb.reshape(-1, 3)[win_src]
    valid[win_tgt] = True
    zbuf = np.zeros(H * W, dtype=np.float64)
    zbuf[win_tgt] = z[win_src]
    out = out.reshape(H, W, 3)
    valid = valid.reshape(H, W)
    zbuf = zbuf.reshape(H, W)

    # Forward splatting leaves 1-px cracks where the surface is stretched; close them with
    # a 3x3 nearest-valid fill that never invents content beyond one pixel.
    holes = ~valid
    if holes.any():
        pad_img = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        pad_val = np.pad(valid, 1, mode="constant")
        filled = out.copy()
        zfill = zbuf.copy()
        pad_z = np.pad(zbuf, 1, mode="edge")
        newly = np.zeros_like(valid)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb_val = pad_val[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
                take = holes & nb_val & ~newly
                if take.any():
                    filled[take] = pad_img[1 + dr:1 + dr + H, 1 + dc:1 + dc + W][take]
                    zfill[take] = pad_z[1 + dr:1 + dr + H, 1 + dc:1 + dc + W][take]
                    newly |= take
        out, valid, zbuf = filled, valid | newly, zfill
    if return_depth:
        return out, valid, zbuf
    return out, valid


def _sample_consistent(img, depth, r, c, z, tol):
    """Bilinear colour at float pixel-centre coords (r, c), using only the corners whose
    depth agrees with the expected depth z (so thin structures and depth edges do not mix
    foreground and background, and occluded points are rejected). Returns (colour, ok)."""
    H, W = depth.shape
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    fr, fc = r - r0, c - c0
    acc = np.zeros((len(r), img.shape[2]), np.float32)
    wsum = np.zeros(len(r), np.float32)
    for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
        ri = np.clip(r0 + dr, 0, H - 1)
        ci = np.clip(c0 + dc, 0, W - 1)
        good = np.abs(depth[ri, ci] - z) < tol
        wk = np.where(good, w + 1e-6, 0.0).astype(np.float32)
        acc += wk[:, None] * img[ri, ci].astype(np.float32)
        wsum += wk
    ok = wsum > 0
    return acc / np.maximum(wsum, 1e-12)[:, None], ok


def _splat_depth(depth, f_src, T_dst_src, f_dst, max_fp=5):
    """Z-buffered depth of the source surface seen from dst, each source point covering a
    footprint as wide as its local magnification (f_dst z_src / f_src z_dst), so a near
    surface leaves no cracks through which the background behind it could bleed."""
    H, W = depth.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = depth.reshape(-1)
    ok = (d > 1e-4) & (d < 10.0)
    x = (cc.reshape(-1) + 0.5 - W / 2.0) * d / f_src
    y = (rr.reshape(-1) + 0.5 - H / 2.0) * d / f_src
    p = T_dst_src @ np.stack([x, y, -d, np.ones_like(d)])
    z = -p[2]
    ok &= z > 1e-4
    u = f_dst * p[0] / np.maximum(z, 1e-9) + W / 2.0          # continuous col (pixel edges at ints)
    v = f_dst * p[1] / np.maximum(z, 1e-9) + H / 2.0
    half = 0.5 * np.clip(f_dst * d / (f_src * np.maximum(z, 1e-9)), 1.0, max_fp) + 0.05
    idx = np.flatnonzero(ok & (u > -max_fp) & (u < W + max_fp) & (v > -max_fp) & (v < H + max_fp))
    u, v, z, half = u[idx], v[idx], z[idx], half[idx]
    zbuf = np.full(H * W, np.inf)
    r0, c0 = np.floor(v).astype(np.int64), np.floor(u).astype(np.int64)
    R = int(np.ceil(max_fp / 2)) + 1
    for dr in range(-R, R + 1):
        rt = r0 + dr
        in_r = (rt >= 0) & (rt < H) & (np.abs(rt + 0.5 - v) <= half)
        if not in_r.any():
            continue
        for dc in range(-R, R + 1):
            ct = c0 + dc
            sel = in_r & (ct >= 0) & (ct < W) & (np.abs(ct + 0.5 - u) <= half)
            if sel.any():
                np.minimum.at(zbuf, rt[sel] * W + ct[sel], z[sel])
    zbuf = zbuf.reshape(H, W)
    valid = np.isfinite(zbuf)
    return np.where(valid, zbuf, 0.0), valid


def inverse_warp(
    rgb: np.ndarray,
    depth: np.ndarray,
    f_px: float,
    T_world_src: np.ndarray,
    T_world_dst: np.ndarray,
    f_dst: float | None = None,
    fill: int = 128,
    fill_rgb: np.ndarray | None = None,
    fill_mask: np.ndarray | None = None,
    grow: int = 1,
    occ_tol: float = 0.02,
    occ_rel: float = 0.03,
    inpaint_iters: int = 2,
):
    """Re-render `rgb` from camera `dst`: forward-splat DEPTH, backward-sample COLOUR.

    Forward splatting alone leaves a stipple of cracks wherever the destination magnifies
    the source (the camera moved away, or the view is foreshortened), because each source
    pixel lands on at most one destination pixel. Here only the depth is splatted, each
    point covering a footprint as wide as its local magnification (remaining 1-px gaps are
    closed by growing up to `grow` pixels, background-preferring), and every destination
    pixel then samples the source colour bilinearly. A pixel is kept
    only if the source actually sees that 3D point (its source depth agrees within
    occ_tol + occ_rel * z), so grown depth never paints across disocclusions.

    Holes that remain -- parts of the destination view the source camera never saw -- are
    filled from `fill_rgb` where `fill_mask` allows (e.g. the empty-scene calibration
    reference, which shows the static workspace from exactly this viewpoint), else `fill`.
    Returns (warped rgb, valid mask) where valid marks pixels seen by the source camera.
    """
    H, W = depth.shape
    fd = f_px if f_dst is None else f_dst
    z, known = _splat_depth(depth, f_px, np.linalg.inv(T_world_dst) @ T_world_src, fd)
    for _ in range(grow):
        holes = ~known
        if not holes.any():
            break
        pz = np.pad(z, 1, mode="constant")
        best = np.zeros_like(z)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    best = np.maximum(best, pz[1 + dr:1 + dr + H, 1 + dc:1 + dc + W])
        take = holes & (best > 0)
        z[take] = best[take]
        known |= take

    rr, cc = np.nonzero(known)
    zz = z[rr, cc]
    x = (cc + 0.5 - W / 2.0) * zz / fd
    y = (rr + 0.5 - H / 2.0) * zz / fd
    p = np.linalg.inv(T_world_src) @ T_world_dst @ np.stack([x, y, -zz, np.ones_like(zz)])
    zs = -p[2]
    cs = f_px * p[0] / np.maximum(zs, 1e-9) + W / 2.0 - 0.5   # float col of pixel centres
    rs = f_px * p[1] / np.maximum(zs, 1e-9) + H / 2.0 - 0.5
    ok = (zs > 1e-4) & (cs >= -0.5) & (cs <= W - 0.5) & (rs >= -0.5) & (rs <= H - 0.5)
    col, good = _sample_consistent(rgb, depth, rs[ok], cs[ok], zs[ok], occ_tol + occ_rel * zs[ok])
    ok[np.flatnonzero(ok)[~good]] = False
    col = col[good]

    out = np.empty((H, W, 3), np.uint8)
    out[:] = fill
    seen = np.zeros((H, W), bool)
    out[rr[ok], cc[ok]] = np.clip(np.round(col), 0, 255).astype(np.uint8)
    seen[rr[ok], cc[ok]] = True
    # isolated holes (speckle along depth edges under strong magnification): average of the
    # seen 3x3 neighbours, only where most of the neighbourhood is seen
    filled = seen.copy()
    img = out.astype(np.float32)
    for _ in range(inpaint_iters):
        holes = ~filled
        if not holes.any():
            break
        pv = np.pad(filled, 1).astype(np.float32)
        pi = np.pad(img * filled[..., None], ((1, 1), (1, 1), (0, 0)))
        cnt = np.zeros((H, W), np.float32)
        acc = np.zeros((H, W, 3), np.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    cnt += pv[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
                    acc += pi[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        take = holes & (cnt >= 5)
        img[take] = acc[take] / cnt[take][:, None]
        filled |= take
    out = np.clip(np.round(img), 0, 255).astype(np.uint8)
    out[~filled] = fill
    if fill_rgb is not None:
        use = ~filled if fill_mask is None else (~filled & fill_mask)
        out[use] = fill_rgb[use]
    return out, seen
