"""
Corrected checkpoint loader + smoke test for the veRL dry run's real
saved checkpoint - reuses the SAME persistent Modal Volume
(grpo-vlm-verl-checkpoints), no new training run needed.

Root cause established by scripts/inspect_checkpoint_raw.py (2026-08-10):
training itself was healthy (LoRA A/B matrices in the raw FSDP .pt
checkpoint have small, sane, NaN-free values consistent with 3 real
tiny-lr gradient steps) - the actual bug is in how the checkpoint gets
loaded. veRL's "huggingface" export still carries PEFT's wrapper prefix
on every key (base_model.model.model.*), confirmed directly (a probe
tensor lookup failed with "no matching reference tensor found" against
a plain Qwen2_5_VLForConditionalGeneration state dict). Loading that
export with a plain from_pretrained() silently drops almost every real
weight (name mismatch), leaving the model near its random initialization
- which explains BOTH the gibberish output AND the "vision changed"
finding (random-init vision weights trivially differ from pretrained
ones) via one mechanism, not two separate bugs.

The fix: reconstruct the EXACT SAME PEFT wrapping veRL used during
training (LoraConfig with target_modules="all-linear",
exclude_modules=".*visual.*" - the real veRL CLI flags from
src/training/dry_run_config.py, not this project's own
build_lora_target_modules() helper, which was for the earlier raw-HF
proxy path and uses different path assumptions), load the RAW FSDP .pt
checkpoint's state dict into that correctly-wrapped model (matching key
structure, not guessed), then merge_and_unload() for clean generation.

This script both builds that corrected loader AND runs the exact same
two real checks as before (vision_unchanged, real completions) so the
result is directly comparable to the earlier (failed) attempts.

Usage: modal run scripts/verify_checkpoint_correct_loading.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

CHECKPOINT_DIR = "/checkpoints"
checkpoint_volume = modal.Volume.from_name("grpo-vlm-verl-checkpoints", create_if_missing=False)

LORA_RANK = 64
LORA_ALPHA = 32
N_SANITY_PROMPTS = 5


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_CACHE_DIR: model_cache, CHECKPOINT_DIR: checkpoint_volume},
    timeout=20 * 60,
)
def verify_corrected_loading() -> dict:
    import os

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.controls.manipulation_check import check_vision_unchanged, hash_vision_state_dict
    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt

    result = {"steps": []}

    # ---- Find the real raw checkpoint file (same paths already confirmed real) ----
    raw_ckpt_candidates = []
    for root, _dirs, files in os.walk(CHECKPOINT_DIR):
        for f in files:
            if f.startswith("model_world_size") and f.endswith(".pt"):
                raw_ckpt_candidates.append(os.path.join(root, f))
    if not raw_ckpt_candidates:
        result["status"] = "FAIL"
        result["error"] = "No raw FSDP checkpoint found"
        return result
    raw_ckpt_path = sorted(raw_ckpt_candidates)[-1]
    result["steps"].append(f"Using raw checkpoint: {raw_ckpt_path}")

    # ---- Load fresh base model, hash its vision weights BEFORE any wrapping ----
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=MODEL_CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    base_vision_hashes = hash_vision_state_dict(base_model)
    result["steps"].append(f"Fresh base model loaded, hashed {len(base_vision_hashes)} vision params")

    problems = load_gsm8k("test")[:N_SANITY_PROMPTS]

    def generate_all(model) -> list[str]:
        outs = []
        for ex in problems:
            prompt_text = build_text_prompt(ex.question)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
            decoded = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            outs.append(decoded)
        return outs

    base_completions = generate_all(base_model)
    result["base_completions"] = [
        {"question": p.question, "ground_truth": p.final_answer, "completion": c}
        for p, c in zip(problems, base_completions)
    ]
    result["steps"].append("Generated real base-model completions (for before/after comparison)")

    # ---- Reconstruct the EXACT SAME PEFT wrapping veRL used during training ----
    # Real flags from src/training/dry_run_config.py, not this project's own
    # build_lora_target_modules() helper (different code path, different
    # key-naming assumptions - verified NOT to apply here).
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        exclude_modules=".*visual.*",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    result["steps"].append(
        f"Reconstructed PEFT wrapping (r={LORA_RANK}, alpha={LORA_ALPHA}, "
        f"target_modules=all-linear, exclude_modules=.*visual.*)"
    )

    # ---- Load the raw checkpoint's real state dict into this correctly-wrapped model ----
    raw_state = torch.load(raw_ckpt_path, map_location="cpu", weights_only=False)
    model_state = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    del raw_state

    load_result = peft_model.load_state_dict(model_state, strict=False)
    result["missing_keys_count"] = len(load_result.missing_keys)
    result["unexpected_keys_count"] = len(load_result.unexpected_keys)
    result["missing_keys_sample"] = load_result.missing_keys[:15]
    result["unexpected_keys_sample"] = load_result.unexpected_keys[:15]
    result["steps"].append(
        f"Loaded checkpoint into correctly-wrapped model: "
        f"{len(load_result.missing_keys)} missing keys, {len(load_result.unexpected_keys)} unexpected keys"
    )

    # ---- Merge LoRA into base weights for clean generation (real PEFT API) ----
    peft_model = peft_model.to("cuda")
    merged_model = peft_model.merge_and_unload()
    result["steps"].append("merge_and_unload() completed")

    trained_completions = generate_all(merged_model)
    result["trained_completions"] = [
        {"question": p.question, "ground_truth": p.final_answer, "completion": c}
        for p, c in zip(problems, trained_completions)
    ]
    result["steps"].append("Generated real completions from the correctly-loaded trained model")

    trained_vision_hashes = hash_vision_state_dict(merged_model)
    vision_check = check_vision_unchanged(base_vision_hashes, trained_vision_hashes)
    result["vision_unchanged"] = vision_check
    result["steps"].append(f"Vision-unchanged check (corrected loading): {vision_check['unchanged']}")

    result["status"] = "PASS" if vision_check["unchanged"] else "FAIL"
    return result


@app.local_entrypoint()
def run():
    result = verify_corrected_loading.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    print(f"\nMissing keys: {result['missing_keys_count']} (sample: {result['missing_keys_sample']})")
    print(f"Unexpected keys: {result['unexpected_keys_count']} (sample: {result['unexpected_keys_sample']})")
    print(f"\nvision_unchanged: {result['vision_unchanged']}")

    print("\n=== BEFORE vs AFTER completions (coherence check) ===")
    for base_c, trained_c in zip(result["base_completions"], result["trained_completions"]):
        print(f"\nQuestion: {base_c['question'][:150]}...")
        print(f"Ground truth: {base_c['ground_truth']}")
        print(f"BASE:    {base_c['completion'][:300]}")
        print(f"TRAINED: {trained_c['completion'][:300]}")

    print(f"\nOverall status: {result['status']}")
