"""Memory-lean optimisers for full fine-tuning of pi05 with DDP.

FSDP is not an option for LeRobot's pi05: `compute_layer_complete` calls the attention
projections, MLP and norms of each decoder layer directly instead of calling the layer's
`forward`, so FSDP units wrapped at the decoder layer never all-gather their parameters.

`adamw_zero1` keeps full bf16 parameters and gradients on every rank (so pi05's forward is
untouched) and shards only the AdamW state across ranks with PyTorch's
`ZeroRedundancyOptimizer` (ZeRO stage 1). With 4 ranks this saves 3/4 of the optimiser
state -- about 10 GB per GPU for a 3.3B model held in bf16.

Enable it for any pi05-family policy with `POSESHIFT_OPTIM=zero1` in the environment.
"""

from __future__ import annotations

import collections
import os
from dataclasses import asdict, dataclass

import torch
import torch.distributed.optim  # noqa: F401  (ZeroRedundancyOptimizer)
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig


def _groups_without_moments(param_groups) -> list[dict]:
    return [
        {k: v for k, v in g.items() if k != "params"} | {"params": list(range(len(g["params"])))}
        for g in param_groups
    ]


class _ShardedAdamW(torch.optim.Optimizer):
    """ZeRO-1 AdamW over parameters of mixed dtype.

    pi05 keeps its vision tower and every RMSNorm in fp32 and the rest in bf16, while
    `ZeroRedundancyOptimizer` requires one dtype. We shard each dtype group with its own
    ZeRO instance and expose the union of their `param_groups` (the same dict objects), so
    LeRobot's LR scheduler and grad clipping work unchanged.

    `state_dict()` returns only hyper-parameters: LeRobot writes the optimiser from the main
    process alone, whereas ZeRO would need every rank to join a consolidation collective.
    A resumed run therefore restarts the Adam moments (logged); model weights are unaffected.
    """

    def __init__(self, params, **adam_kwargs):
        from torch.distributed.optim import ZeroRedundancyOptimizer

        params = [p for p in params if p.requires_grad]
        by_dtype: dict[torch.dtype, list] = {}
        for p in params:
            by_dtype.setdefault(p.dtype, []).append(p)
        self._inner = [
            ZeroRedundancyOptimizer(ps, optimizer_class=torch.optim.AdamW, **adam_kwargs)
            for _, ps in sorted(by_dtype.items(), key=lambda kv: str(kv[0]))
        ]
        # Minimal Optimizer state without registering our own copy of the parameters.
        self.defaults = dict(adam_kwargs)
        self.state = collections.defaultdict(dict)
        self.param_groups = [g for o in self._inner for g in o.param_groups]
        for name in (
            "_optimizer_step_pre_hooks", "_optimizer_step_post_hooks",
            "_optimizer_state_dict_pre_hooks", "_optimizer_state_dict_post_hooks",
            "_optimizer_load_state_dict_pre_hooks", "_optimizer_load_state_dict_post_hooks",
        ):
            setattr(self, name, collections.OrderedDict())
        self._zero_grad_profile_name = "Optimizer.zero_grad#_ShardedAdamW.zero_grad"
        if torch.distributed.get_rank() == 0:
            sizes = {str(dt): sum(p.numel() for p in ps) for dt, ps in by_dtype.items()}
            print(f"[poseshift.optim] ZeRO-1 AdamW over {len(self._inner)} dtype groups: {sizes}", flush=True)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = closure() if closure is not None else None
        for o in self._inner:
            o.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):  # noqa: D102
        for o in self._inner:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):  # noqa: D102
        return {"state": {}, "param_groups": _groups_without_moments(self.param_groups)}

    def load_state_dict(self, state_dict):  # noqa: D102
        print("[poseshift.optim] restoring hyper-parameters only; Adam moments restart", flush=True)
        for g, sg in zip(self.param_groups, state_dict.get("param_groups", []), strict=False):
            for k, v in sg.items():
                if k != "params":
                    g[k] = v


@OptimizerConfig.register_subclass("adamw_zero1")
@dataclass
class AdamWZero1Config(AdamWConfig):
    def build(self, params):
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        dist = torch.distributed
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return torch.optim.AdamW(params, **kwargs)
        return _ShardedAdamW(params, **kwargs)


def maybe_patch_pi05_preset() -> None:
    """Swap the AdamW preset of every pi05-family config for ZeRO-1 when requested."""
    if os.environ.get("POSESHIFT_OPTIM", "").lower() != "zero1":
        return
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if getattr(PI05Config, "_poseshift_zero1", False):
        return
    original = PI05Config.get_optimizer_preset

    def get_optimizer_preset(self):
        base = original(self)
        return AdamWZero1Config(**{k: getattr(base, k) for k in ("lr", "betas", "eps", "weight_decay", "grad_clip_norm")})

    PI05Config.get_optimizer_preset = get_optimizer_preset
    PI05Config._poseshift_zero1 = True


def maybe_patch_ddp_bucket_view() -> None:
    """Make accelerate's DDP reuse gradient storage for its all-reduce buckets.

    LeRobot builds `DistributedDataParallelKwargs(find_unused_parameters=True)`; DDP then keeps
    a second, bucketed copy of every gradient (another 6.6 GB for pi05 in bf16). With
    `gradient_as_bucket_view=True` the buckets alias the `.grad` tensors instead.
    Enabled with `POSESHIFT_DDP_BUCKET_VIEW=1`.
    """
    if os.environ.get("POSESHIFT_DDP_BUCKET_VIEW", "") != "1":
        return
    from accelerate.utils import DistributedDataParallelKwargs as K

    if getattr(K, "_poseshift_bucket_view", False):
        return
    original_init = K.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("gradient_as_bucket_view", True)
        original_init(self, *args, **kwargs)
        if os.environ.get("RANK", "0") == "0":
            print(f"[poseshift.optim] DDP gradient_as_bucket_view={self.gradient_as_bucket_view}", flush=True)

    K.__init__ = __init__
    K._poseshift_bucket_view = True


def maybe_patch_freeze_token_embedding() -> None:
    """Freeze PaliGemma's 257k x 2048 token-embedding table (527M params).

    LIBERO prompts use a few hundred distinct tokens, yet the embedding receives a dense
    gradient and a full set of Adam moments (~3 GB per GPU here). Every transformer block,
    the vision tower and the action expert are still trained in full.
    Enabled with `POSESHIFT_FREEZE_EMBED=1`.
    """
    if os.environ.get("POSESHIFT_FREEZE_EMBED", "") != "1":
        return
    from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel as M

    if getattr(M, "_poseshift_freeze_embed", False):
        return
    original = M._set_requires_grad

    def _set_requires_grad(self):
        original(self)
        emb = self.paligemma.get_input_embeddings()
        for p in emb.parameters():
            p.requires_grad = False
        if os.environ.get("RANK", "0") == "0" and not getattr(self, "_poseshift_logged", False):
            n = sum(p.numel() for p in emb.parameters())
            print(f"[poseshift.optim] froze token embedding ({n / 1e6:.0f}M params)", flush=True)
            self._poseshift_logged = True

    M._set_requires_grad = _set_requires_grad
    M._poseshift_freeze_embed = True
