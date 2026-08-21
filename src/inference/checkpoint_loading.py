"""
Verified checkpoint loading, extracted so every downstream evaluation
shares ONE code path.

WHY THIS MODULE EXISTS
----------------------
scripts/run_phase1_eval_on_modal.py contains a carefully-built loading
routine whose docstring documents the trap it exists to avoid: veRL's
`global_step_N/actor/huggingface/` export still carries PEFT's wrapper
prefix, so a plain `from_pretrained()` silently drops almost every
trained weight and yields something numerically ~identical to the BASE
model. That failure is silent - it produces Delta ~= 0 with no error
anywhere, which looks exactly like a real null result.

Phase 1b adds three more evaluations that must each load a trained
checkpoint (trajectory sweep, random-reward control, harder-dataset
eval). Copy-pasting the loading routine into each would create four
independent chances to reintroduce that silent bug, and - worse - four
code paths that could drift apart, so that two evaluations of the "same"
checkpoint quietly disagree. This project has already been bitten once by
exactly that class of error (the extraction regex shared between reward
and eval, commit 43279ab).

So: one implementation, imported everywhere. Any future fix lands in all
callers at once.

The logic here is a faithful extraction of the routine proven on the real
Phase 1 run (commit bda1c14), generalised only in that the checkpoint
step is a parameter rather than a module constant.
"""

from __future__ import annotations

import hashlib

DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_TARGET_MODULES = "all-linear"
DEFAULT_EXCLUDE_MODULES = ".*visual.*"


def load_base_model(model_id: str, cache_dir: str):
    """Untrained base model + processor, bf16, on CPU (caller moves to device)."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    return model, processor


def hash_vision_weights(model) -> str:
    """
    SHA-256 over the vision tower's parameters (PLAN.md Phase 1 item 8).

    Training was text-only and excluded `.*visual.*` from LoRA, so this
    hash MUST match between base and any trained checkpoint. A mismatch
    means the central manipulation claim - "the vision pathway is
    untouched" - is false, and the experiment is not what it says it is.
    """
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if "visual" in name:
            h.update(name.encode())
            h.update(param.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def _unwrap_dtensors(model_state: dict) -> tuple[dict, int]:
    """
    FSDP2 saves DTensor objects, not plain tensors, and load_state_dict
    fails with "aten.copy_.default got mixed torch.Tensor and DTensor".

    Training ran with trainer.n_gpus_per_node=1, so each DTensor has
    exactly one shard and `.to_local()` IS the full tensor - no gather and
    no process group required. This assumption is checked, not assumed:
    a DTensor with more than one shard would silently yield a PARTIAL
    weight here, so we refuse rather than load a fraction of a tensor.
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # older torch layout
        from torch.distributed._tensor import DTensor

    n = sum(1 for v in model_state.values() if isinstance(v, DTensor))
    if n == 0:
        return model_state, 0

    for key, value in model_state.items():
        if isinstance(value, DTensor):
            local, full = value.to_local().shape, value.shape
            if tuple(local) != tuple(full):
                raise RuntimeError(
                    f"DTensor {key!r} is genuinely sharded (local={tuple(local)}, "
                    f"full={tuple(full)}). This loader assumes world_size=1, where "
                    f"to_local() is the whole tensor. Loading would silently use a "
                    f"PARTIAL weight - refusing."
                )
            break  # one representative check is enough for a uniform save

    return {k: (v.to_local() if isinstance(v, DTensor) else v) for k, v in model_state.items()}, n


def load_rl_checkpoint(
    model,
    checkpoint_path: str,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    target_modules: str = DEFAULT_TARGET_MODULES,
    exclude_modules: str = DEFAULT_EXCLUDE_MODULES,
    verbose: bool = True,
):
    """
    Rebuild training's PEFT wrapping around `model` and load the raw FSDP
    state dict into it, then merge the adapter down.

    `model` must be a freshly-loaded BASE model. The LoRA config must
    match training exactly - a mismatch in rank/alpha/target_modules
    produces adapter keys that do not line up, which `strict=False` would
    then swallow silently.

    Raises rather than returns a base-model-lookalike if no LoRA tensors
    are present, which is the specific silent failure this whole module
    exists to prevent.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    del raw_state

    model_state, n_dtensors = _unwrap_dtensors(model_state)
    if verbose and n_dtensors:
        print(f"unwrapped {n_dtensors} DTensor -> Tensor (FSDP2 checkpoint, world_size=1)", flush=True)

    n_lora = sum(1 for k in model_state if "lora_" in k)
    if n_lora == 0:
        raise RuntimeError(
            f"No LoRA tensors found in {checkpoint_path} - refusing to run an evaluation that "
            f"would silently measure the base model and report it as the RL model."
        )

    result = peft_model.load_state_dict(model_state, strict=False)
    if verbose:
        print(
            f"loaded state dict: {len(result.missing_keys)} missing, "
            f"{len(result.unexpected_keys)} unexpected, {n_lora} lora tensors present",
            flush=True,
        )
    del model_state

    merged = peft_model.merge_and_unload()
    if verbose:
        print("merge_and_unload() done", flush=True)
    return merged


def download_checkpoint(
    repo_id: str,
    step: int,
    hf_token: str,
    cache_dir: str,
    filename_template: str = "global_step_{step}/actor/model_world_size_1_rank_0.pt",
) -> str:
    """
    Fetch one checkpoint's raw FSDP state dict from the HF Hub mirror.

    The Hub copy, not a Modal Volume, is the source of truth: Volumes are
    workspace-scoped and this project has already moved Modal accounts
    more than once (the Phase 1 eval records live on a different account
    than the one that will run Phase 1b). The Hub copy survives that.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename_template.format(step=step),
        token=hf_token,
        cache_dir=cache_dir,
    )


def local_checkpoint_path(
    checkpoint_dir: str,
    step: int,
    filename_template: str = "global_step_{step}/actor/model_world_size_1_rank_0.pt",
) -> str:
    """
    Path to a checkpoint already present on a mounted Modal Volume.

    Used by the Phase 1b random-reward control, whose checkpoints never
    leave the account that trains them - so a Hub round-trip would be
    pure cost. The Hub mirror exists to survive ACCOUNT switches (see
    download_checkpoint); within one account a Volume is the direct path.
    """
    import os

    path = os.path.join(checkpoint_dir, filename_template.format(step=step))
    if not os.path.exists(path):
        available = []
        if os.path.isdir(checkpoint_dir):
            available = sorted(
                d for d in os.listdir(checkpoint_dir) if d.startswith("global_step_")
            )
        raise FileNotFoundError(
            f"No checkpoint at {path}. Available in {checkpoint_dir}: {available or '(none)'}"
        )
    return path


def load_model_at_step(
    model_id: str,
    cache_dir: str,
    step: int | None,
    repo_id: str | None = None,
    hf_token: str | None = None,
    local_checkpoint_dir: str | None = None,
    device: str = "cuda",
    verbose: bool = True,
    **lora_kwargs,
):
    """
    One call for "give me the model as it was at training step N".

    step=None means the untrained base model. Otherwise the checkpoint is
    resolved from `local_checkpoint_dir` if given (a mounted Volume), else
    downloaded from the Hub mirror `repo_id`. Returns
    (model, processor, vision_sha256) with the model on `device` and in
    eval mode.

    The vision hash is returned on EVERY load, not just the RL one, so
    callers can assert base/RL equality without a second forward pass -
    the manipulation check becomes free and automatic rather than a step
    someone has to remember.
    """
    model, processor = load_base_model(model_id, cache_dir)

    if step is not None:
        if local_checkpoint_dir:
            path = local_checkpoint_path(local_checkpoint_dir, step)
            if verbose:
                print(f"loading global_step_{step} from local volume {path}", flush=True)
        else:
            if not repo_id or not hf_token:
                raise ValueError(
                    "Provide either local_checkpoint_dir, or repo_id + hf_token, "
                    "to load a trained checkpoint."
                )
            if verbose:
                print(f"downloading global_step_{step} from {repo_id} ...", flush=True)
            path = download_checkpoint(repo_id, step, hf_token, cache_dir)
        model = load_rl_checkpoint(model, path, verbose=verbose, **lora_kwargs)

    model = model.to(device).eval()
    vision_sha = hash_vision_weights(model)
    if verbose:
        label = "base" if step is None else f"step_{step}"
        print(f"[{label}] vision-tower sha256: {vision_sha}", flush=True)
    return model, processor, vision_sha
