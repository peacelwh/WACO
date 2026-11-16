# Data

This folder holds the small files that define the splits of the paper; the larger assets are
downloaded next to them by `bash scripts/download_assets.sh`. No private or sensitive data
is stored here.

| File | Content | Source |
| --- | --- | --- |
| `fewshot_episodes_10.json`, `fewshot_episodes_20.json` | episode indices of the first 10 / 20 demonstrations of every task of `lerobot/libero` (400 / 800 episodes; the 10-shot set is a subset of the 20-shot set) | `python scripts/fewshot_episodes.py DATASET_ROOT K` |
| `liberoplus_camera_ids.json` | task ids of the LIBERO-Plus camera track per suite (376 + 396 + 408 + 419 = 1,599 variants) | `python scripts/liberoplus_camera_ids.py` |
| `calib_libero/<suite>__<task>.npz` *(downloaded)* | calibration bundle of every LIBERO task (40 tasks): empty-workspace RGB-D frame from the nominal camera, static-scene mask, camera-to-base extrinsic, nominal base pose, focal length, instruction | `python scripts/build_calibration.py` |
| `calib_liberoplus/<suite>__<task>.npz` *(downloaded)* | calibration bundles recorded in the LIBERO-Plus scenes, with the twelve-view scan of the empty workspace | `python scripts/build_calibration_liberoplus.py` |
| `depth_dav2_128.npy`, `depth_dav2_128.json` *(downloaded with `WITH_DEPTH=1`)* | per-frame metric depth of the 273,465 third-person frames of `lerobot/libero` (Depth Anything V2, aligned to the calibration scale; float16, 128 x 128), used by the round-trip reprojection of the error-aware fine-tuning | `python scripts/precompute_depth.py` |

Training demonstrations are the public LIBERO demonstrations in LeRobot format
([`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero)), used without modification:
50 demonstrations per task, collected from one camera pose and one base pose.

All released assets are hosted at
[`Bluexxx/LIBERO-PoseShift`](https://huggingface.co/datasets/Bluexxx/LIBERO-PoseShift).
