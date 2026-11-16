"""Patch LeRobot's training script: micro-batch gradient accumulation in `update_policy`.

Why: full fine-tuning of pi05 (3.3B parameters, bf16 with an fp32 vision tower) leaves room
for only a few samples of activations per GPU. Splitting each per-GPU batch into
micro-batches keeps one optimiser update per training step (so the global batch of 32, the
LR schedule and the step count are unchanged) while bounding activation memory.

Enabled only when `POSESHIFT_MICRO_BS` > 0 and smaller than the per-GPU batch; otherwise the
original code path runs byte-for-byte. Idempotent: re-running is a no-op.
"""

import sys
from pathlib import Path

if len(sys.argv) > 1:
    TARGET = Path(sys.argv[1])
else:  # the installed LeRobot
    import importlib.util

    TARGET = Path(importlib.util.find_spec("lerobot.scripts.lerobot_train").origin)
src = TARGET.read_text()
MARK = "# [poseshift] micro-batch accumulation"
if MARK in src:
    print("already patched")
    sys.exit(0)

helpers = f'''

{MARK} -------------------------------------------------------------
def _poseshift_batch_size(batch):
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.ndim > 0:
            return int(v.shape[0])
    return 0


def _poseshift_split_batch(batch, micro_bs):
    n = _poseshift_batch_size(batch)
    chunks = []
    for s in range(0, n, micro_bs):
        e = min(n, s + micro_bs)
        chunk = {{}}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == n:
                chunk[k] = v[s:e]
            elif isinstance(v, (list, tuple)) and len(v) == n:
                chunk[k] = type(v)(v[s:e])
            else:
                chunk[k] = v
        chunks.append((chunk, e - s))
    return chunks
# ---------------------------------------------------------------------------


def update_policy('''
src = src.replace("\n\ndef update_policy(", helpers, 1)

old = '''        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)
'''
new = '''        elif not _poseshift_micro:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    if _poseshift_micro:  # [poseshift] micro-batch accumulation, one optimiser update per step
        chunks = _poseshift_split_batch(batch, _poseshift_micro_bs)
        loss = torch.zeros((), device=accelerator.device)
        for i, (chunk, n) in enumerate(chunks):
            sync_ctx = accelerator.no_sync(policy) if i < len(chunks) - 1 else nullcontext()
            with sync_ctx:
                with accelerator.autocast():
                    micro_loss, output_dict = policy.forward(chunk)
                accelerator.backward(micro_loss * (n / _poseshift_bsz))
            loss = loss + micro_loss.detach() * (n / _poseshift_bsz)
    else:
        # Use accelerator's backward method
        accelerator.backward(loss)
'''
assert old in src, "update_policy body changed upstream; refusing to patch"
src = src.replace(old, new, 1)

old2 = '''    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if sample_weights is not None:'''
new2 = '''    _poseshift_micro_bs = int(os.environ.get("POSESHIFT_MICRO_BS", "0") or 0)
    _poseshift_bsz = _poseshift_batch_size(batch)
    _poseshift_micro = sample_weights is None and 0 < _poseshift_micro_bs < _poseshift_bsz

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if sample_weights is not None:'''
assert old2 in src
src = src.replace(old2, new2, 1)

if "\nimport os\n" not in src:
    src = src.replace("\nimport time\n", "\nimport os\nimport time\n", 1)
TARGET.write_text(src)
print("patched", TARGET)
