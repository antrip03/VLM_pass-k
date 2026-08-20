"""
Phase 1 evaluation (PLAN.md Section 3 Phase 1 items 7-9): T/D/E sampling
for the base and RL checkpoints, pass@k, and bootstrap CIs on
Delta_text / Delta_pixel - the numbers that actually answer this
project's research question.

Runs the two models as two SEPARATE, parallel Modal calls rather than one
sequential job: halves wall-clock (~4hr instead of ~8hr on A10G at the
measured 301s/problem), and isolates failure - if one model's run dies,
the other's records still land on the Volume and only the failed half
needs re-running.

CHECKPOINT LOADING - the trap this script exists to avoid: veRL's
"huggingface" export under global_step_N/actor/huggingface/ still carries
PEFT's wrapper prefix, so loading it with a plain from_pretrained()
silently drops almost every trained weight and yields something
numerically ~identical to the base model (documented in
scripts/verify_checkpoint_correct_loading.py, which was written
specifically to catch this). That failure is silent and would look like a
real null result - Delta ~= 0 with no error anywhere. The correct,
verified path is used here instead: load the base model, re-wrap it in
the SAME LoraConfig training used, load the raw FSDP state dict
(model_world_size_1_rank_0.pt) with strict=False, then merge_and_unload().

Checkpoints are pulled from the HF Hub mirror rather than a Modal Volume,
because the Volume is workspace-scoped and this project has already
switched Modal accounts twice - the Hub copy is the one that survives
that (src/training/hf_checkpoint_sync.py).

Usage:
    modal run scripts/run_phase1_eval_on_modal.py
    modal run scripts/run_phase1_eval_on_modal.py --n 128 --problems 50
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

RESULTS_DIR = "/results"
results_volume = modal.Volume.from_name("grpo-vlm-phase1-eval-results", create_if_missing=True)

HF_CKPT_REPO = "GunGG4/grpo-vlm-phase1-checkpoints"
CKPT_STEP = "global_step_467"
LORA_RANK = 32
LORA_ALPHA = 32
EVAL_GPU = "A10G"  # measured as fast as A100 here at n=128, at ~half the rate


def _load_base(torch, MODEL_CACHE_DIR):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=MODEL_CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    return model, processor


def _hash_vision_weights(model) -> str:
    """PLAN.md Phase 1 item 8 (manipulation check): the vision tower must be
    byte-identical before/after training - training was text-only and
    excluded .*visual.* from LoRA, so any change here means the experiment
    is not what it claims to be."""
    import hashlib

    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if "visual" in name:
            h.update(name.encode())
            h.update(param.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


@app.function(
    image=image,
    gpu=EVAL_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, RESULTS_DIR: results_volume},
    timeout=20 * 60 * 60,
    secrets=[modal.Secret.from_dict({})],  # replaced per-call with the real HF token
)
def eval_one_model(model_kind: str, n: int, problems: int, image_chunk: int) -> dict:
    """model_kind: "base" or "rl"."""
    import json
    import os
    import time

    os.environ["HF_HOME"] = MODEL_CACHE_DIR

    import torch

    from src.data.gsm8k_loader import load_gsm8k
    from src.inference.run_sampling import run_sampling, save_sampling_records

    print(f"=== eval_one_model(kind={model_kind}, n={n}, problems={problems}) ===", flush=True)
    t0 = time.time()
    model, processor = _load_base(torch, MODEL_CACHE_DIR)
    print(f"base weights loaded in {time.time() - t0:.1f}s", flush=True)

    if model_kind == "rl":
        # See module docstring: the huggingface/ export is NOT safe to
        # from_pretrained() directly. Reconstruct training's PEFT wrapping
        # and load the raw FSDP state dict into it.
        from huggingface_hub import hf_hub_download
        from peft import LoraConfig, get_peft_model

        print(f"downloading {CKPT_STEP} raw state dict from {HF_CKPT_REPO} ...", flush=True)
        raw_path = hf_hub_download(
            repo_id=HF_CKPT_REPO,
            filename=f"{CKPT_STEP}/actor/model_world_size_1_rank_0.pt",
            token=os.environ["HF_TOKEN"],
            cache_dir=MODEL_CACHE_DIR,
        )
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules="all-linear",
            exclude_modules=".*visual.*",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_config)
        raw_state = torch.load(raw_path, map_location="cpu", weights_only=False)
        model_state = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
        del raw_state

        # DTensor unwrapping (2026-08-20) - a direct, downstream
        # consequence of this project's own FSDP2 fix. FSDP2 shards
        # per-parameter via DTensor, so its saved state dict holds DTensor
        # objects, not plain tensors, and load_state_dict fails with
        # "aten.copy_.default got mixed torch.Tensor and DTensor". The
        # pattern this loading path was modelled on
        # (scripts/verify_checkpoint_correct_loading.py) predates the FSDP2
        # switch and was written against FSDP1's flat plain tensors.
        # Training ran with trainer.n_gpus_per_node=1, so each DTensor has
        # exactly one shard and to_local() IS the full tensor - no gather,
        # no process group needed.
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:  # older torch layout
            from torch.distributed._tensor import DTensor

        n_dtensors = sum(1 for v in model_state.values() if isinstance(v, DTensor))
        if n_dtensors:
            model_state = {
                k: (v.to_local() if isinstance(v, DTensor) else v) for k, v in model_state.items()
            }
            print(f"unwrapped {n_dtensors} DTensor -> Tensor (FSDP2 checkpoint, world_size=1)", flush=True)

        # Validate BEFORE loading, so a checkpoint with no adapter weights
        # is reported as such rather than surfacing as a confusing load error.
        n_lora_in_ckpt = sum(1 for k in model_state if "lora_" in k)
        if n_lora_in_ckpt == 0:
            raise RuntimeError(
                "No LoRA tensors found in the checkpoint state dict - refusing to run an "
                "evaluation that would silently measure the base model and report it as the "
                "RL model (see this module's docstring)."
            )

        load_result = peft_model.load_state_dict(model_state, strict=False)
        print(f"loaded state dict: {len(load_result.missing_keys)} missing, "
              f"{len(load_result.unexpected_keys)} unexpected, {n_lora_in_ckpt} lora tensors present",
              flush=True)
        del model_state
        model = peft_model.merge_and_unload()
        print("merge_and_unload() done", flush=True)

    model = model.to("cuda").eval()
    vision_hash = _hash_vision_weights(model)
    print(f"vision-tower sha256: {vision_hash}", flush=True)

    examples = load_gsm8k("test")[:problems]
    records = run_sampling(
        model, processor, examples, conditions=["T", "D", "E"], n=n,
        progress_label=model_kind, image_chunk=image_chunk,
    )

    out_path = f"{RESULTS_DIR}/records_{model_kind}.parquet"
    save_sampling_records(records, out_path)
    results_volume.commit()

    # Truncation-rate check (PLAN.md Phase 1 item 7, mandatory): a sample
    # with no extracted answer is the symptom to watch, per condition.
    trunc = {}
    for cond in ["T", "D", "E"]:
        sub = [r for r in records if r.condition == cond]
        trunc[cond] = round(sum(1 for r in sub if r.extracted_answer is None) / max(len(sub), 1), 4)

    summary = {
        "model_kind": model_kind,
        "n": n,
        "problems": len(examples),
        "records": len(records),
        "elapsed_hr": round((time.time() - t0) / 3600, 2),
        "vision_sha256": vision_hash,
        "no_answer_rate": trunc,
        "accuracy": {
            cond: round(
                sum(1 for r in records if r.condition == cond and r.correct)
                / max(sum(1 for r in records if r.condition == cond), 1),
                4,
            )
            for cond in ["T", "D", "E"]
        },
        "out_path": out_path,
    }
    with open(f"{RESULTS_DIR}/summary_{model_kind}.json", "w") as f:
        json.dump(summary, f, indent=2)
    results_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=60 * 60)
def analyze() -> dict:
    """pass@k + paired bootstrap CIs on Delta_text / Delta_pixel."""
    import json

    from src.inference.run_sampling import load_sampling_records
    from src.metrics.bootstrap_ci import bootstrap_delta_ci, bootstrap_pass_at_k_ci

    dfs = {kind: load_sampling_records(f"{RESULTS_DIR}/records_{kind}.parquet") for kind in ("base", "rl")}

    def per_problem(df, condition):
        """(n, c) per problem, ordered by problem_idx so base/rl stay paired."""
        sub = df[df["condition"] == condition]
        g = sub.groupby("problem_idx")["correct"]
        return [(int(cnt), int(corr)) for cnt, corr in zip(g.count(), g.sum())]

    out = {"pass_at_k": {}, "deltas": {}}
    for condition in ["T", "D", "E"]:
        for k in (1, 64):
            for kind in ("base", "rl"):
                res = bootstrap_pass_at_k_ci(per_problem(dfs[kind], condition), k=k)
                out["pass_at_k"][f"{kind}/{condition}/k={k}"] = {
                    "mean": round(res["point_estimate"], 4),
                    "ci": [round(res["ci_lower"], 4), round(res["ci_upper"], 4)],
                }
            d = bootstrap_delta_ci(per_problem(dfs["rl"], condition), per_problem(dfs["base"], condition), k=k)
            out["deltas"][f"Delta_{condition}/k={k}"] = {
                "delta": round(d["point_estimate"], 4),
                "ci": [round(d["ci_lower"], 4), round(d["ci_upper"], 4)],
                "significant": d["significant"],
            }

    # PLAN.md's headline quantities, named as the plan names them.
    out["headline"] = {
        "Delta_text (condition T, k=1)": out["deltas"]["Delta_T/k=1"],
        "Delta_pixel (condition E, k=1)": out["deltas"]["Delta_E/k=1"],
        "Delta_decomposed (condition D, k=1)": out["deltas"]["Delta_D/k=1"],
    }
    with open(f"{RESULTS_DIR}/analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    results_volume.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(n: int = 128, problems: int = 50, image_chunk: int = 32):
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN is required (the RL checkpoint lives on the HF Hub mirror).")

    fn = eval_one_model.with_options(secrets=[modal.Secret.from_dict({"HF_TOKEN": hf_token})])

    print("Spawning base and RL evals in parallel (2 x A10G)...")
    calls = {kind: fn.spawn(kind, n, problems, image_chunk) for kind in ("base", "rl")}
    for kind, call in calls.items():
        print(f"  {kind}: call_id={call.object_id}")
    print("\nBoth running independently - safe to disconnect. Re-attach with the call ids above,")
    print("then run:  modal run scripts/run_phase1_eval_on_modal.py::analyze")
