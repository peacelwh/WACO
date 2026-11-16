"""Accuracy of the WACO front end against the true shifts, and the behaviour of its gate.

Uses the environment exactly as at evaluation time (`canon="waco"`, no homing) and compares
the runtime's estimates with the simulator's true poses:
  dT_c (camera shift), T_bc (camera in base frame), dT_b (base shift, planar).
"""

from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark  # noqa: E402

from poseshift.estimator import pose_error  # noqa: E402
from poseshift.geometry import camera_extrinsics, invert  # noqa: E402
from poseshift.libero_env import LiberoPoseShiftEnv  # noqa: E402

from poseshift.paths import ROOT  # noqa: E402
SUITE = os.environ.get("SUITE", "libero_spatial")
TASKS = [int(t) for t in os.environ.get("TASKS", "0,3,6").split(",")]
N_EP = int(os.environ.get("N_EP", "4"))
LEVELS = os.environ.get("LEVELS", "l1,l2,l3").split(",")
suite = benchmark.get_benchmark_dict()[SUITE]()


def base_pose(sim):
    b = sim.model.body_name2id("robot0_base")
    T = np.eye(4)
    T[:3, :3] = sim.data.body_xmat[b].reshape(3, 3)
    T[:3, 3] = sim.data.body_xpos[b]
    return T


rows = []
for task in TASKS:
    for kind in ("clean", "cam", "base", "both"):
        for level in (["clean"] if kind == "clean" else LEVELS):
            for ep in range(N_EP):
                e = LiberoPoseShiftEnv(task_suite=suite, task_id=task, task_suite_name=SUITE,
                                       obs_type="pixels_agent_pos", observation_height=256, observation_width=256,
                                       episode_index=ep, perturb_kind=kind, perturb_level=level,
                                       action_frame="base", canon="waco", canon_homing=False)
                e.reset(seed=ep)
                sim, rt, cal = e._env.sim, e._waco, e._calibration
                T_wc = invert(camera_extrinsics(sim, "agentview"))
                T_wb = base_pose(sim)
                T_wc0 = cal.T_wb0 @ cal.T_bc0
                gt_dTc = invert(T_wc0) @ T_wc
                gt_dTb = invert(cal.T_wb0) @ T_wb
                gt_Tbc = invert(T_wb) @ T_wc
                rows.append({"task": task, "kind": kind, "level": level, "ep": ep,
                             "dTc": pose_error(rt.dT_c, gt_dTc), "Tbc": pose_error(rt.T_bc, gt_Tbc),
                             "dTb": pose_error(rt.dT_b, gt_dTb),
                             "gt_dTc_mag": pose_error(gt_dTc, np.eye(4)), "gt_dTb_mag": pose_error(gt_dTb, np.eye(4)),
                             "use_cam": rt.use_cam, "use_base": rt.use_base, "src": rt.info.get("scene_src"),
                             **{k: rt.info[k] for k in ("scene_inliers", "icp_rms_mm", "consistent", "offplane_z_mm", "offplane_rp_deg")}})
                e.close()
            sel = [r for r in rows if r["task"] == task and r["kind"] == kind and r["level"] == level]

            def fmt(key):
                t = np.array([r[key][0] for r in sel]) * 1000
                a = np.array([r[key][1] for r in sel])
                return f"{np.median(t):6.1f}mm/{np.median(a):5.2f}deg(max {t.max():6.1f}/{a.max():5.2f})"
            print(f"task{task} {kind:<5} {level:<5} | dT_c {fmt('dTc')} | T_bc {fmt('Tbc')} | dT_b {fmt('dTb')} "
                  f"| inl {np.median([r['scene_inliers'] for r in sel]):.0f}", flush=True)

bad = [r for r in rows if r["dTb"][0] > 0.02 or r["dTb"][1] > 1.5]
print(f"\ncatastrophic (dTb > 2 cm or 1.5 deg): {len(bad)}/{len(rows)}; caught by the consistency gate: "
      f"{sum(not r['consistent'] for r in bad)}; false alarms on good estimates: "
      f"{sum(not r['consistent'] for r in rows if r not in bad)}")
print("\n=== summary over tasks (median / 90th pct) ===")
for kind in ("clean", "cam", "base", "both"):
    for level in (["clean"] if kind == "clean" else LEVELS):
        sel = [r for r in rows if r["kind"] == kind and r["level"] == level]
        out = []
        for key in ("dTc", "dTb"):
            t = np.array([r[key][0] for r in sel]) * 1000
            a = np.array([r[key][1] for r in sel])
            out.append(f"{key} {np.median(t):5.1f}/{np.quantile(t, .9):6.1f} mm  {np.median(a):4.2f}/{np.quantile(a, .9):5.2f} deg")
        print(f"{kind:<5} {level:<5} | " + " | ".join(out), flush=True)

out = os.path.join(ROOT, "outputs", "estimator")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, f"probe_{SUITE}{os.environ.get('OUT_TAG', '')}.json"), "w") as fh:
    json.dump(rows, fh, indent=1, default=float)
print("ESTIMATOR_PROBE_DONE")
