"""Import hook that registers LIBERO-PoseShift and the WACO training options with LeRobot.

Installed as a `.pth` file by `scripts/install_plugin.sh`, so `lerobot-train` and
`lerobot-eval` pick up `--env.type=libero_poseshift` without any edit to LeRobot.
Registration failures are logged, never raised: a broken plugin must not break the host.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from . import env_config  # noqa: F401  (registers libero_poseshift)
except Exception as exc:  # pragma: no cover
    log.debug("poseshift env registration skipped: %r", exc)

try:
    from .optim import maybe_patch_ddp_bucket_view, maybe_patch_freeze_token_embedding, maybe_patch_pi05_preset

    maybe_patch_pi05_preset()
    maybe_patch_ddp_bucket_view()
    maybe_patch_freeze_token_embedding()
except Exception as exc:  # pragma: no cover
    log.warning("poseshift optimiser registration failed: %r", exc)

try:
    from .rt_aug import maybe_patch_dataset

    maybe_patch_dataset()
except Exception as exc:  # pragma: no cover
    log.warning("poseshift error-aware augmentation registration failed: %r", exc)
