"""LeRobot `EnvConfig` for the LIBERO-PoseShift benchmark.

Registered as ``--env.type=libero_poseshift``. It reuses LeRobot's LIBERO config so the
feature map, observation processor (180-degree image rotation, 8-D state) and vectorised
construction are identical to the official benchmark. Only the per-env class changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.envs.configs import EnvConfig, LiberoEnv as LiberoEnvConfig


@EnvConfig.register_subclass("libero_poseshift")
@dataclass
class LiberoPoseShiftEnvConfig(LiberoEnvConfig):
    # 'clean' | 'cam' | 'base' | 'both'
    perturb_kind: str = "clean"
    # 'clean' | 'l1' | 'l2' | 'l3'
    perturb_level: str = "clean"
    perturb_seed: int = 0
    # 'bump': the camera translates and rotates about its own centre (the benchmark's shifts)
    # 'orbit': additionally scales the camera's distance to the scene (not used by the levels)
    perturb_mode: str = "bump"
    # 'base': base-frame protocol (default) | 'world': LIBERO's native world-frame convention
    action_frame: str = "base"
    # GT oracle (upper bound, canonicalises with the true shifts): 'none' | 'cam' | 'base' | 'both'
    oracle_canon: str = "none"
    oracle_homing: bool = True
    # WACO: canonicalisation with shifts estimated at test time: 'none' | 'waco'
    canon: str = "none"
    canon_homing: bool = True

    @property
    def gym_kwargs(self) -> dict:
        kw = dict(super().gym_kwargs)
        kw.update(
            perturb_kind=self.perturb_kind,
            perturb_level=self.perturb_level,
            perturb_seed=self.perturb_seed,
            perturb_mode=self.perturb_mode,
            action_frame=self.action_frame,
            oracle_canon=self.oracle_canon,
            oracle_homing=self.oracle_homing,
            canon=self.canon,
            canon_homing=self.canon_homing,
        )
        return kw

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        """Mirror of `lerobot.envs.libero.create_libero_envs` that builds our env class.

        We do not monkeypatch LeRobot's module: its factories resolve `LiberoEnv` lazily
        (possibly inside a worker, after any temporary patch is undone), and a permanent patch
        would silently change `--env.type=libero` in the same process.
        """
        from collections import defaultdict
        from functools import partial

        import gymnasium as gym
        from lerobot.envs.configs import _make_vec_env_cls
        from lerobot.envs.libero import _get_suite, _select_task_ids
        from lerobot.envs.utils import _LazyAsyncVectorEnv, parse_camera_names

        from .libero_env import LiberoPoseShiftEnv

        if self.task is None:
            raise ValueError("libero_poseshift requires --env.task")
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        gym_kwargs = dict(self.gym_kwargs)
        task_ids_filter = gym_kwargs.pop("task_ids", None)
        camera_names = parse_camera_names(self.camera_name)
        suite_names = [s.strip() for s in str(self.task).split(",") if s.strip()]
        is_async = env_cls is gym.vector.AsyncVectorEnv

        def make_one(suite, suite_name, task_id, episode_index):
            return LiberoPoseShiftEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name=suite_name,
                camera_name=camera_names,
                init_states=self.init_states,
                episode_length=self.episode_length,
                episode_index=episode_index,
                n_envs=n_envs,
                control_mode=self.control_mode,
                camera_name_mapping=self.camera_name_mapping,
                is_libero_plus=self.is_libero_plus,
                **gym_kwargs,
            )

        out: dict = defaultdict(dict)
        for suite_name in suite_names:
            suite = _get_suite(suite_name)
            selected = _select_task_ids(len(suite.tasks), task_ids_filter)
            cached = (None, None, None)
            for tid in selected:
                fns = [partial(make_one, suite, suite_name, tid, ep) for ep in range(n_envs)]
                if is_async:
                    lazy = _LazyAsyncVectorEnv(fns, *cached)
                    if cached[0] is None:
                        cached = (lazy.observation_space, lazy.action_space, lazy.metadata)
                    out[suite_name][tid] = lazy
                else:
                    out[suite_name][tid] = env_cls(fns)
                print(f"[libero_poseshift] suite={suite_name} task={tid} n_envs={n_envs} "
                      f"kind={self.perturb_kind} level={self.perturb_level} frame={self.action_frame}")
        return {k: dict(v) for k, v in out.items()}


__all__ = ["LiberoPoseShiftEnvConfig"]
