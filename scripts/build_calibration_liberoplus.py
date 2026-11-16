"""WACO calibration bundles recorded in the LIBERO-Plus scenes, with a multi-view scan.

LIBERO-Plus ships its own scene assets, so the calibration bundle is recorded in its scenes:
the nominal camera's empty-workspace RGB-D frame plus a short scan of twelve further views
(azimuth +-20, +-45, +-70 degrees at the nominal height, +-45 and +-70 degrees raised by 15
degrees, a view 1.6x farther away, and one raised view), all looking at the same workspace
point, each stored with its pose in the nominal camera frame. A camera that has been orbited
far from the nominal view can then be matched against the nearest scan view. The
canonicalisation target is still the nominal view. On a real robot this is a one-minute
hand-held sweep of the empty table.

Requires the LIBERO bundles (scripts/build_calibration.py) for the nominal camera poses.

  python scripts/build_calibration_liberoplus.py [suite ...]
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from scipy.spatial.transform import Rotation as R  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from poseshift.geometry import camera_extrinsics, invert, scipy2mj  # noqa: E402
from poseshift.libero_env import LiberoPoseShiftEnv  # noqa: E402
from poseshift.waco_runtime import WacoRuntime, arm_mask  # noqa: E402
from poseshift.warp import real_depth  # noqa: E402

from poseshift.paths import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "data", "calib_liberoplus")
os.makedirs(OUT, exist_ok=True)
SUFFIX = re.compile(r"\s+view\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+initstate\s+\d+$")
# scan: ring of azimuths about the vertical axis through the workspace centre, one pulled back
# along the optical axis, one raised
SCAN = [("az", a) for a in (-70, -45, -20, 20, 45, 70)] + [("dist", 1.6), ("el", 15)]
SCAN += [("azel", (a, 15)) for a in (-70, -45, 45, 70)]   # second ring, raised by 15 degrees
SCAN_STATES = (40, 41, 42)   # calibration-only init states, as for the nominal reference


def scan_pose(T_wc0, P, kind, v):
    C0, R0 = T_wc0[:3, 3], T_wc0[:3, :3]
    r = C0 - P
    if kind == "azel":                       # raise first, then swing around the vertical axis
        T_el = scan_pose(T_wc0, P, "el", v[1])
        return scan_pose(T_el, P, "az", v[0])
    if kind == "az":
        Rm = R.from_euler("z", v, degrees=True).as_matrix()
    elif kind == "el":
        h = np.cross([0.0, 0.0, 1.0], r)
        h /= np.linalg.norm(h)
        Rm = R.from_rotvec(h * np.deg2rad(v)).as_matrix()
        if (P + Rm @ r)[2] < C0[2]:          # pick the sign that RAISES the camera
            Rm = R.from_rotvec(-h * np.deg2rad(v)).as_matrix()
    else:
        Rm = np.eye(3)
    T = np.eye(4)
    T[:3, :3] = Rm @ R0
    T[:3, 3] = P + (v if kind == "dist" else 1.0) * (Rm @ r)
    return T


def render_view(e, cal, T_wc, depth_tol=0.002):
    """Empty-scene RGB-D from camera pose T_wc (world), with its static mask."""
    sim, m = e._env.sim, e._env.sim.model
    cid = m.camera_name2id("agentview")
    m.cam_pos[cid] = T_wc[:3, 3]
    m.cam_quat[cid] = scipy2mj(R.from_matrix(T_wc[:3, :3]))
    sim.forward()
    frames = []
    for i in SCAN_STATES:
        e._env.set_init_state(e._init_states[i])
        e._hide_objects()
        frames.append(e._env.env._get_observations(force_update=True))
    depths = np.stack([real_depth(sim, f["agentview_depth"]) for f in frames])
    depth = depths[0]
    static = (depths.std(axis=0) < depth_tol) & (depth > 1e-3) & (depth < 3.0)
    T_cb = invert(invert(cal.T_wb0) @ T_wc)
    not_arm = arm_mask(T_cb, cal.cloud.points_base(sim), cal.f_px, *depth.shape, dilate=8, depth=depth)
    return frames[0]["agentview_image"].copy(), depth.astype(np.float32), static & not_arm


suites = sys.argv[1:] or ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# nominal camera pose of every original task, from the LIBERO calibration bundles
from poseshift.libero_env import _calib_language_index  # noqa: E402
NOMINAL = _calib_language_index(os.path.join(ROOT, "data", "calib_libero"))
for sname in suites:
    su = benchmark.get_benchmark_dict()[sname]()
    names = [su.get_task(i).name for i in range(len(su.tasks))]
    # LIBERO-Plus has no unperturbed copy of a task with the default robot: its nominal-view
    # variants (initstate_k>0) swap the robot's initial configuration. So open the ORIGINAL
    # task's scene through one of its camera variants (initstate_0, default robot), put the
    # camera back at the calibrated nominal pose, and photograph the empty workspace there.
    stems = sorted({n.split("_view_")[0] for n in names if "_view_" in n and "_language_" not in n})
    print(sname, "original tasks:", len(stems), flush=True)
    for k_task, stem in enumerate(stems):
        path = os.path.join(OUT, f"{sname}__{k_task}.npz")
        if os.path.exists(path):
            continue
        tid = next(i for i, n in enumerate(names) if n.startswith(stem + "_view_") and n.endswith("_initstate_0"))
        # LIBERO's own instruction (it strips e.g. the "LIVING_ROOM_SCENE2_" prefix of libero_10
        # names), minus LIBERO-Plus's " view ... initstate k" suffix
        lang = SUFFIX.sub("", su.get_task(tid).language.strip())
        if lang.lower() not in NOMINAL:
            print("  no LIBERO calibration bundle for", lang, flush=True)
            continue
        z = np.load(NOMINAL[lang.lower()], allow_pickle=True)
        T_wc_nom = z["T_wb0"] @ z["T_bc0"]
        e = LiberoPoseShiftEnv(task_suite=su, task_id=tid, task_suite_name=sname, obs_type="pixels_agent_pos",
                               observation_height=256, observation_width=256, need_depth=True, is_libero_plus=True)
        e.reset(seed=0)
        sim = e._env.sim
        cid = sim.model.camera_name2id("agentview")
        sim.model.cam_pos[cid] = T_wc_nom[:3, 3]
        sim.model.cam_quat[cid] = scipy2mj(R.from_matrix(T_wc_nom[:3, :3]))
        sim.forward()
        cal = WacoRuntime.calibrate(sim, e.empty_scene_frames(), "agentview")
        T_wc0 = invert(camera_extrinsics(sim, "agentview"))
        H, W = cal.ref_depth.shape
        c = cal.ref_depth[H // 2 - 4:H // 2 + 4, W // 2 - 4:W // 2 + 4]
        P = T_wc0[:3, 3] - T_wc0[:3, 2] * float(np.median(c[(c > 1e-3) & (c < 3.0)]))   # looks down -z
        vT, vrgb, vdep, vmask = [], [], [], []
        for kind, v in SCAN:
            T_wc = scan_pose(T_wc0, P, kind, v)
            rgb, dep, msk = render_view(e, cal, T_wc)
            vT.append(invert(T_wc0) @ T_wc)
            vrgb.append(rgb)
            vdep.append(dep)
            vmask.append(msk)
        np.savez_compressed(path, ref_rgb=cal.ref_rgb, ref_depth=cal.ref_depth.astype(np.float32),
                            ref_mask=cal.ref_mask, T_wb0=cal.T_wb0, T_bc0=cal.T_bc0, f_px=cal.f_px, language=lang,
                            views_T=np.stack(vT), views_rgb=np.stack(vrgb), views_depth=np.stack(vdep),
                            views_mask=np.stack(vmask), views_kind=np.array([f"{k}{v}" for k, v in SCAN]))
        drift = np.linalg.norm(T_wc0[:3, 3] - T_wc_nom[:3, 3])
        print(f"{sname}[{k_task}] via task {tid} static={cal.ref_mask.mean():.2f} "
              f"scan static={np.mean([m.mean() for m in vmask]):.2f} cam drift {1000 * drift:.1f} mm :: {lang}", flush=True)
        e.close()
print("CALIBRATION_LP_DONE")
