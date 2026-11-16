"""One-off setup for every LIBERO task: the WACO calibration bundle.

Per task (nominal camera, nominal base; init states 40..49, which are never evaluated):
  ref_rgb, ref_depth   RGB-D frame of the EMPTY workspace from the nominal camera (raw MuJoCo
                       orientation; the movable objects are removed)
  ref_mask             static-scene mask (projected arm and invalid depth removed)
  T_wb0, T_bc0, f_px   nominal base pose, camera-to-base extrinsic, focal length in pixels
  language             task instruction, used to map LeRobot dataset tasks to bundles

Written to data/calib_libero/<suite>__<task_id>.npz and consumed by the error-aware
augmentation (training) and by the WACO runtime (evaluation).
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark  # noqa: E402

from poseshift.libero_env import LiberoPoseShiftEnv  # noqa: E402

from poseshift.paths import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "data", "calib_libero")
os.makedirs(OUT, exist_ok=True)

for sname in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    su = benchmark.get_benchmark_dict()[sname]()
    for tid in range(len(su.tasks)):
        path = os.path.join(OUT, f"{sname}__{tid}.npz")
        if os.path.exists(path):
            continue
        e = LiberoPoseShiftEnv(task_suite=su, task_id=tid, task_suite_name=sname, obs_type="pixels_agent_pos",
                               observation_height=256, observation_width=256, need_depth=True)
        e.reset(seed=0)
        from poseshift.waco_runtime import WacoRuntime
        cal = WacoRuntime.calibrate(e._env.sim, e.empty_scene_frames(), "agentview")
        np.savez_compressed(path, ref_rgb=cal.ref_rgb, ref_depth=cal.ref_depth.astype(np.float32),
                            ref_mask=cal.ref_mask, T_wb0=cal.T_wb0, T_bc0=cal.T_bc0, f_px=cal.f_px,
                            language=su.get_task(tid).language)
        print(f"{sname}[{tid}] static={cal.ref_mask.mean():.2f} :: {su.get_task(tid).language}", flush=True)
        e.close()
print("CALIBRATION_DONE")
