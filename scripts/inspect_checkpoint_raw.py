"""
Direct, low-level inspection of the checkpoint saved by the last real
dry run (scripts/run_verl_dry_run_on_modal.py) - reuses the SAME
persistent Modal Volume (grpo-vlm-verl-checkpoints), no new training
run needed, since the goal is to understand what ALREADY got saved, not
re-produce it.

Two real, unresolved findings motivate this:
  1. vision_unchanged=False after the fix believed to address it
     (use_orig_params=True) - the fix did not work, falsified by direct
     evidence, not assumed.
  2. Reading veRL's actual base_model_merger.py source (2026-08-10)
     revealed save_hf_model_and_tokenizer() explicitly POPS LoRA
     parameters out of the state_dict before saving - meaning the
     "hf_model" checkpoint we loaded and generated from previously was
     pure base-model weights with NO LoRA applied at all. That's
     confusing on its own: loading pure base weights should behave
     identically to loading the base model fresh, yet one produced
     coherent text and the "trained" one produced garbage in the same
     test - pointing at either (a) base weights genuinely got corrupted/
     modified somewhere before or during save (consistent with the
     vision-changed finding), or (b) a save/reconstruction bug scrambling
     how FSDP-sharded weights get gathered back into a single state_dict,
     unrelated to LoRA at all.

This script does NOT trust any documentation's claimed directory layout
(the official checkpoint docs describe the Megatron backend's structure,
we're using FSDP - a real, live mismatch caught before writing code
against the wrong assumption) - it lists the REAL directory tree first,
then inspects real tensors directly (raw statistics: NaN/Inf, mean, std,
max-abs-diff against a freshly-downloaded pretrained reference), rather
than going through model.generate() again (too many intermediate steps
that could mask or introduce their own issues).

Usage: modal run scripts/inspect_checkpoint_raw.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

CHECKPOINT_DIR = "/checkpoints"
checkpoint_volume = modal.Volume.from_name("grpo-vlm-verl-checkpoints", create_if_missing=False)


@app.function(
    image=image,
    gpu="A100-40GB",  # loading the checkpoint safetensors (~8.4GB), the raw
    # .pt FSDP checkpoint (~8.4GB), and a fresh reference model (~7.5GB)
    # together would be tight on a 24GB A10G - real headroom matters here
    # since this is a one-shot diagnostic, not worth fighting memory over.
    volumes={MODEL_CACHE_DIR: model_cache, CHECKPOINT_DIR: checkpoint_volume},
    timeout=15 * 60,
)
def inspect_raw() -> dict:
    import os

    import torch

    result = {"steps": []}

    # ---- Step 1: real directory tree, not assumed ----
    # Keep path and size SEPARATE (a first version of this script
    # concatenated them into one display string and then tried to
    # pattern-match on that same string for logic - a real bug in this
    # diagnostic itself, caught by its own output: the path-detection
    # queries below returned empty despite the files clearly being
    # present in the tree listing).
    tree = []  # real file paths only, for logic
    tree_display = []  # path + size, for human-readable printing only
    for root, dirs, files in os.walk(CHECKPOINT_DIR):
        depth = root.replace(CHECKPOINT_DIR, "").count(os.sep)
        if depth > 4:
            continue
        for f in files:
            full = os.path.join(root, f)
            size_mb = os.path.getsize(full) / 1e6
            tree.append(full)
            tree_display.append(f"{full} ({size_mb:.2f} MB)")
    result["directory_tree"] = tree_display
    result["steps"].append(f"Found {len(tree)} files under {CHECKPOINT_DIR}")

    # ---- Step 2: find and load the actual hf_model checkpoint's raw weights ----
    # Real, confirmed paths from the first (buggy-detection) run's tree
    # listing, not guessed: .../actor/huggingface/model.safetensors and
    # .../actor/model_world_size_1_rank_0.pt (the raw FSDP-format
    # checkpoint) and .../actor/lora_train_meta.json.
    hf_dirs = sorted({os.path.dirname(f) for f in tree if os.path.basename(f) == "config.json"})
    result["hf_dirs_found"] = hf_dirs
    result["steps"].append(f"Candidate HF checkpoint dirs: {hf_dirs}")

    raw_ckpt_files = [f for f in tree if f.endswith(".pt") and "model_world_size" in os.path.basename(f)]
    result["raw_checkpoint_files_found"] = raw_ckpt_files
    result["steps"].append(f"Raw FSDP checkpoint files: {raw_ckpt_files}")

    lora_meta_files = [f for f in tree if os.path.basename(f) == "lora_train_meta.json"]
    result["lora_meta_files_found"] = lora_meta_files
    if lora_meta_files:
        with open(lora_meta_files[0]) as f:
            result["lora_train_meta_content"] = f.read()
        result["steps"].append(f"lora_train_meta.json content: {result['lora_train_meta_content']}")

    if not hf_dirs:
        result["status"] = "FAIL"
        result["error"] = "No HF checkpoint directory found - cannot proceed with tensor inspection"
        return result

    checkpoint_path = hf_dirs[0]

    # ---- Step 4: load raw state dict directly (not via model.generate) ----
    from safetensors.torch import load_file

    safetensor_files = [
        os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
    ]
    result["safetensor_files"] = safetensor_files
    result["steps"].append(f"Found {len(safetensor_files)} safetensors files in checkpoint")

    checkpoint_state = {}
    for sf in safetensor_files:
        checkpoint_state.update(load_file(sf))
    result["steps"].append(f"Loaded {len(checkpoint_state)} tensors from checkpoint")

    # ---- Step 5: real statistics on the raw checkpoint tensors ----
    nan_tensors = []
    inf_tensors = []
    sample_stats = {}
    for name, tensor in checkpoint_state.items():
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan:
            nan_tensors.append(name)
        if has_inf:
            inf_tensors.append(name)

    result["n_tensors_total"] = len(checkpoint_state)
    result["n_tensors_with_nan"] = len(nan_tensors)
    result["n_tensors_with_inf"] = len(inf_tensors)
    result["nan_tensor_names_sample"] = nan_tensors[:10]
    result["inf_tensor_names_sample"] = inf_tensors[:10]
    result["steps"].append(
        f"Checkpoint tensors: {len(checkpoint_state)} total, "
        f"{len(nan_tensors)} contain NaN, {len(inf_tensors)} contain Inf"
    )

    # ---- Step 6: load a FRESH reference base model, compare real tensors ----
    from transformers import Qwen2_5_VLForConditionalGeneration

    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=MODEL_CACHE_DIR
    )
    ref_state = ref_model.state_dict()
    result["steps"].append(f"Reference base model loaded: {len(ref_state)} tensors")

    # Compare a handful of real, specific tensors: a vision tensor, a
    # language decoder tensor, and the embedding - real names, not guessed
    # blindly, cross-checked against what's actually in checkpoint_state.
    candidate_names = [
        n
        for n in checkpoint_state
        if ("visual.patch_embed" in n)
        or ("layers.0.self_attn.q_proj.weight" in n)
        or ("embed_tokens.weight" in n)
    ][:5]
    comparisons = []
    for name in candidate_names:
        ckpt_tensor = checkpoint_state[name].float()
        ref_name = name if name in ref_state else name.replace("model.", "", 1)
        if ref_name not in ref_state:
            comparisons.append({"name": name, "error": "no matching reference tensor found"})
            continue
        ref_tensor = ref_state[ref_name].float()
        if ckpt_tensor.shape != ref_tensor.shape:
            comparisons.append(
                {"name": name, "error": f"shape mismatch: ckpt={tuple(ckpt_tensor.shape)} ref={tuple(ref_tensor.shape)}"}
            )
            continue
        max_abs_diff = (ckpt_tensor - ref_tensor).abs().max().item()
        rel_diff = max_abs_diff / (ref_tensor.abs().max().item() + 1e-8)
        comparisons.append(
            {
                "name": name,
                "checkpoint_stats": {
                    "mean": ckpt_tensor.mean().item(),
                    "std": ckpt_tensor.std().item(),
                    "max_abs": ckpt_tensor.abs().max().item(),
                },
                "reference_stats": {
                    "mean": ref_tensor.mean().item(),
                    "std": ref_tensor.std().item(),
                    "max_abs": ref_tensor.abs().max().item(),
                },
                "max_abs_diff_from_reference": max_abs_diff,
                "relative_diff": rel_diff,
            }
        )
    result["tensor_comparisons"] = comparisons
    result["steps"].append(f"Compared {len(comparisons)} real tensors against fresh reference weights")

    # ---- Step 7: the decisive check - inspect the RAW FSDP .pt checkpoint ----
    # This is what actually tells training-corruption and export-corruption
    # apart. veRL's save_hf_model_and_tokenizer() pops LoRA params before
    # writing the "huggingface" export (confirmed by reading its real
    # source earlier) - so the safetensors checkpoint just inspected above
    # is BASE WEIGHTS ONLY, no LoRA. If THIS raw .pt file (which should
    # still hold everything, LoRA included, in FSDP's native save format)
    # shows sane values, training was fine and the export step is where
    # something breaks. If it's ALSO corrupted, the problem predates
    # export entirely.
    del checkpoint_state, ref_state, ref_model
    torch.cuda.empty_cache()

    if raw_ckpt_files:
        raw_state = torch.load(raw_ckpt_files[0], map_location="cpu", weights_only=False)
        result["steps"].append(
            f"Raw .pt checkpoint loaded: top-level type={type(raw_state).__name__}, "
            f"keys={list(raw_state.keys())[:20] if isinstance(raw_state, dict) else 'N/A'}"
        )

        # The raw save format is not guaranteed to be a flat {name: tensor}
        # dict (FSDP checkpoints are often {"model": {...}, "optimizer":
        # {...}, ...} or similar nested structure) - inspect what's
        # actually there rather than assume a shape.
        raw_tensors = {}
        if isinstance(raw_state, dict):
            model_part = raw_state.get("model", raw_state)
            if isinstance(model_part, dict):
                for k, v in model_part.items():
                    if torch.is_tensor(v):
                        raw_tensors[k] = v

        result["raw_checkpoint_tensor_count"] = len(raw_tensors)
        result["steps"].append(f"Extracted {len(raw_tensors)} real tensors from raw checkpoint")

        lora_tensor_names = [n for n in raw_tensors if "lora_" in n.lower()]
        result["raw_lora_tensor_names_sample"] = lora_tensor_names[:10]
        result["raw_lora_tensor_count"] = len(lora_tensor_names)
        result["steps"].append(f"Found {len(lora_tensor_names)} LoRA-named tensors in the raw checkpoint")

        raw_nan = [n for n, t in raw_tensors.items() if torch.isnan(t.float()).any().item()]
        raw_inf = [n for n, t in raw_tensors.items() if torch.isinf(t.float()).any().item()]
        result["raw_n_nan"] = len(raw_nan)
        result["raw_n_inf"] = len(raw_inf)
        result["steps"].append(f"Raw checkpoint: {len(raw_nan)} tensors with NaN, {len(raw_inf)} with Inf")

        # Real LoRA weight statistics - after 3 tiny (lr=3e-6) steps, a
        # LoRA B matrix (zero-initialized by PEFT convention) should have
        # moved only a SMALL amount from exactly zero, not be huge or NaN.
        lora_stats = []
        for name in lora_tensor_names[:10]:
            t = raw_tensors[name].float()
            lora_stats.append(
                {
                    "name": name,
                    "mean": t.mean().item(),
                    "std": t.std().item(),
                    "max_abs": t.abs().max().item(),
                    "has_nan": torch.isnan(t).any().item(),
                }
            )
        result["raw_lora_stats_sample"] = lora_stats
        result["status"] = (
            "PASS"
            if len(nan_tensors) == 0 and len(inf_tensors) == 0 and len(raw_nan) == 0 and len(raw_inf) == 0
            else "FAIL"
        )
    else:
        result["steps"].append("No raw .pt checkpoint file found - cannot check LoRA weights directly")
        result["status"] = "PASS" if len(nan_tensors) == 0 and len(inf_tensors) == 0 else "FAIL"

    return result


@app.local_entrypoint()
def run():
    result = inspect_raw.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    print("\n--- Directory tree (first 60 entries) ---")
    for f in result["directory_tree"][:60]:
        print(f"  {f}")
    if "error" in result:
        print(f"\nERROR: {result['error']}")
    else:
        print(f"\nNaN tensors: {result['n_tensors_with_nan']} / {result['n_tensors_total']}")
        print(f"Inf tensors: {result['n_tensors_with_inf']} / {result['n_tensors_total']}")
        if result["nan_tensor_names_sample"]:
            print(f"Sample NaN tensor names: {result['nan_tensor_names_sample']}")
        print("\n--- Tensor comparisons vs. fresh reference weights (HF export, LoRA-stripped) ---")
        for c in result["tensor_comparisons"]:
            print(f"  {c}")
        if "raw_checkpoint_tensor_count" in result:
            print(f"\n--- Raw FSDP .pt checkpoint (should still contain LoRA) ---")
            print(f"Total tensors: {result['raw_checkpoint_tensor_count']}")
            print(f"LoRA-named tensors found: {result['raw_lora_tensor_count']}")
            print(f"Sample LoRA tensor names: {result['raw_lora_tensor_names_sample']}")
            print(f"NaN: {result['raw_n_nan']}, Inf: {result['raw_n_inf']}")
            print("Real LoRA tensor statistics (should be small after 3 tiny-lr steps):")
            for s in result["raw_lora_stats_sample"]:
                print(f"  {s}")
    if "lora_train_meta_content" in result:
        print(f"\nlora_train_meta.json: {result['lora_train_meta_content']}")
    print(f"\nStatus: {result['status']}")
