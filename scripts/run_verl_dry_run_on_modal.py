"""
Orchestrates the real, tiny veRL Dr. GRPO dry run (PLAN.md Phase 0 item
4) end to end on Modal: prepare a small real data slice -> run
src/training/run_dry_run.sh through the ACTUAL veRL trainer (not a raw-HF
proxy) -> independently verify the result, not just trust veRL's own
logs. Three separate Modal functions, run in sequence:

  1. prepare_data: writes a small real GSM8K slice (enough to cover
     train_batch_size x total_training_steps, with margin) to a Volume,
     using this project's own prepare_data.py (validated locally already).
  2. run_dry_run: runs the actual veRL trainer via subprocess inside the
     real veRL Modal environment (configs/modal_app_verl.py, already
     validated separately). Captures stdout/stderr so real errors are
     visible, not swallowed.
  3. inspect_checkpoint: the independent verification step the user asked
     for explicitly - loads the base model AND the freshly-trained
     checkpoint, generates real completions from BOTH on the same real
     GSM8K problems (so a human can directly compare before/after and
     confirm the trained model is still producing coherent text, not
     gibberish from a bug), and runs the same manipulation-check logic
     already used elsewhere in this project (src/controls/
     manipulation_check.py) against the real trained checkpoint's vision
     weights - not assumed clean, checked.

VERIFICATION STATUS (2026-08-11): a first version of inspect_checkpoint
loaded veRL's "huggingface" export directly via plain
Qwen2_5_VLForConditionalGeneration.from_pretrained() and found
vision_unchanged=False plus gibberish completions - alarming, but a real
bug in the LOADING, not training. Confirmed via
scripts/inspect_checkpoint_raw.py: the raw FSDP checkpoint's LoRA
weights are small, sane, and NaN-free (exactly consistent with 3 real
tiny-lr gradient steps), but veRL's "huggingface" export still carries
PEFT's wrapper prefix (base_model.model.model.*) on every key - a plain
from_pretrained() silently drops nearly all of them (name mismatch),
leaving the model near-random. Fixed here (and confirmed via
scripts/verify_checkpoint_correct_loading.py, real result: 0 missing/0
unexpected keys, vision_unchanged=True across all 390 params, coherent
completions) by loading the RAW checkpoint into a freshly-reconstructed
PEFT wrapping (the exact same LoraConfig veRL used:
target_modules="all-linear", exclude_modules=".*visual.*") instead of
trusting the "huggingface" export directly.

Usage: modal run scripts/run_verl_dry_run_on_modal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

DATA_DIR = "/data"
CHECKPOINT_DIR = "/checkpoints"
data_volume = modal.Volume.from_name("grpo-vlm-verl-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("grpo-vlm-verl-checkpoints", create_if_missing=True)

N_TRAIN_EXAMPLES = 50  # comfortably more than train_batch_size(4) x total_training_steps(3)
N_VAL_EXAMPLES = 20
N_SANITY_PROMPTS = 5  # how many real completions to show before/after for the coherence check

TOTAL_TRAINING_STEPS = 3


@app.function(image=image, volumes={DATA_DIR: data_volume}, timeout=5 * 60)
def prepare_data() -> dict:
    import pandas as pd

    from src.training.prepare_data import build_verl_rows

    train_rows = build_verl_rows("train")[:N_TRAIN_EXAMPLES]
    val_rows = build_verl_rows("test")[:N_VAL_EXAMPLES]
    pd.DataFrame(train_rows).to_parquet(f"{DATA_DIR}/train.parquet")
    pd.DataFrame(val_rows).to_parquet(f"{DATA_DIR}/test.parquet")
    data_volume.commit()
    return {"train_rows": len(train_rows), "val_rows": len(val_rows)}


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_CACHE_DIR: model_cache, DATA_DIR: data_volume, CHECKPOINT_DIR: checkpoint_volume},
    timeout=40 * 60,
)
def run_dry_run() -> dict:
    import os
    import subprocess

    import src as _src  # resolve the REAL mounted path, not a guessed one

    from src.training.dry_run_config import build_args

    src_root = Path(_src.__file__).resolve().parent
    reward_fn_path = src_root / "training" / "reward_fn.py"

    args = build_args(
        model_path=MODEL_ID,
        train_file=f"{DATA_DIR}/train.parquet",
        val_file=f"{DATA_DIR}/test.parquet",
        reward_fn_path=str(reward_fn_path),
        checkpoint_dir=CHECKPOINT_DIR,
        total_training_steps=TOTAL_TRAINING_STEPS,
    )

    env = os.environ.copy()
    env["HF_HOME"] = MODEL_CACHE_DIR
    result = subprocess.run(
        ["python3", "-m", "verl.trainer.main_ppo", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=35 * 60,
    )
    checkpoint_volume.commit()
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-8000:],
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_CACHE_DIR: model_cache, CHECKPOINT_DIR: checkpoint_volume},
    timeout=20 * 60,
)
def inspect_checkpoint() -> dict:
    import glob
    import os

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.controls.manipulation_check import check_vision_unchanged, hash_vision_state_dict
    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt

    result = {"steps": []}

    # Real raw checkpoint layout: {default_local_dir}/global_step_{N}/actor/model_world_size_*.pt
    # NOT the "huggingface" export - see the module docstring's
    # VERIFICATION STATUS note for why that export can't be trusted with a
    # plain from_pretrained() load.
    raw_ckpt_candidates = sorted(
        glob.glob(f"{CHECKPOINT_DIR}/global_step_*/actor/model_world_size*.pt")
    )
    if not raw_ckpt_candidates:
        result["status"] = "FAIL"
        result["error"] = f"No raw FSDP checkpoint found under {CHECKPOINT_DIR}"
        result["dir_listing"] = glob.glob(f"{CHECKPOINT_DIR}/**", recursive=True)[:200]
        return result
    checkpoint_path = raw_ckpt_candidates[-1]
    result["steps"].append(f"Using raw checkpoint: {checkpoint_path}")

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=MODEL_CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    result["steps"].append("Base model loaded")

    base_vision_hashes = hash_vision_state_dict(base_model)
    result["steps"].append(f"Hashed {len(base_vision_hashes)} base-model vision params")

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
    result["steps"].append(f"Generated {len(base_completions)} base-model completions")

    # Reconstruct the EXACT SAME PEFT wrapping veRL used during training
    # (real flags from src/training/dry_run_config.py), then load the raw
    # checkpoint's real state dict into it - not this project's own
    # build_lora_target_modules() helper, a different code path with
    # different key-naming assumptions, verified not to apply here.
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules="all-linear",
        exclude_modules=".*visual.*",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)
    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    del raw_state
    load_result = peft_model.load_state_dict(model_state, strict=False)
    result["missing_keys_count"] = len(load_result.missing_keys)
    result["unexpected_keys_count"] = len(load_result.unexpected_keys)
    result["steps"].append(
        f"Loaded checkpoint into correctly-wrapped model: "
        f"{len(load_result.missing_keys)} missing, {len(load_result.unexpected_keys)} unexpected keys"
    )

    peft_model = peft_model.to("cuda")
    trained_model = peft_model.merge_and_unload()
    result["steps"].append("Trained checkpoint loaded (raw + reconstructed PEFT wrap + merge_and_unload)")

    trained_vision_hashes = hash_vision_state_dict(trained_model)
    vision_check = check_vision_unchanged(base_vision_hashes, trained_vision_hashes)
    result["vision_unchanged"] = vision_check
    result["steps"].append(f"Vision-unchanged check: {vision_check['unchanged']}")

    trained_completions = generate_all(trained_model)
    result["trained_completions"] = [
        {"question": p.question, "ground_truth": p.final_answer, "completion": c}
        for p, c in zip(problems, trained_completions)
    ]
    result["steps"].append(f"Generated {len(trained_completions)} trained-checkpoint completions")

    result["status"] = (
        "PASS"
        if vision_check["unchanged"] and result["missing_keys_count"] == 0 and result["unexpected_keys_count"] == 0
        else "FAIL"
    )
    return result


@app.local_entrypoint()
def main():
    print("=== Step 1: prepare_data ===")
    data_result = prepare_data.remote()
    print(f"  train_rows={data_result['train_rows']}, val_rows={data_result['val_rows']}")

    print("\n=== Step 2: run_dry_run (real veRL trainer) ===")
    train_result = run_dry_run.remote()
    print(f"  returncode: {train_result['returncode']}")
    print("  --- stdout tail ---")
    print(train_result["stdout_tail"])
    print("  --- stderr tail ---")
    print(train_result["stderr_tail"])

    if train_result["returncode"] != 0:
        print("\nTraining FAILED - skipping checkpoint inspection.")
        return

    print("\n=== Step 3: inspect_checkpoint (independent verification) ===")
    inspect_result = inspect_checkpoint.remote()
    for step in inspect_result["steps"]:
        print(f"  - {step}")

    if "error" in inspect_result:
        print(f"\nERROR: {inspect_result['error']}")
        return

    print(
        f"\nMissing keys: {inspect_result['missing_keys_count']}, "
        f"Unexpected keys: {inspect_result['unexpected_keys_count']}"
    )
    print(f"vision_unchanged: {inspect_result['vision_unchanged']}")

    print("\n=== BEFORE vs AFTER completions (coherence check) ===")
    for base_c, trained_c in zip(inspect_result["base_completions"], inspect_result["trained_completions"]):
        print(f"\nQuestion: {base_c['question'][:150]}...")
        print(f"Ground truth: {base_c['ground_truth']}")
        print(f"BASE:    {base_c['completion'][:300]}")
        print(f"TRAINED: {trained_c['completion'][:300]}")

    print(f"\nOverall status: {inspect_result['status']}")
