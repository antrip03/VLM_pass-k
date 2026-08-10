"""
Real Unsloth VRAM/timing smoke test, built to be directly comparable to
the already-measured veRL-path proxy numbers (scripts/vram_smoke_test_2000cap.py,
scripts/timing_calibration_2000cap.py): rollout = 864.8 tok/s aggregate at
64 concurrent sequences; training = 0.48s per micro-batch-of-4 step, with
micro-batch=4 the confirmed ceiling (8/16/32/64 all OOM) on a single 40GB
A100. Same model, same GPU, same ~2000-token target length, same
micro-batch bisection methodology - an apples-to-apples measurement
instead of trusting Unsloth's own published "1.5-2x faster, 90% less
VRAM" benchmark, which is not necessarily measured on this model/workload/
hardware.

Deliberately NOT using load_in_4bit (Unsloth's own RL docs example shows
it, but quantization would introduce a second confound on top of the
tooling question actually being asked - is any speedup from Unsloth's
kernels, or from quantization). bf16, matching every other LoRA config in
this project.

finetune_vision_layers=False is used - and per Unsloth's own docs, this
isn't just this project's preference, it's REQUIRED: "vLLM does not
support LoRA for vision/encoder layers", so Unsloth's fast_inference=True
(vLLM-backed rollout) structurally cannot train vision LoRA adapters.
Convenient alignment with this project's text-only-RL requirement, but
still independently verified here the same way as everywhere else in
this project (src/controls/manipulation_check.py's whole reason for
existing) - not just trusted from the docs.

Usage: modal run scripts/unsloth_smoke_test_2000cap.py
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app_unsloth import GPU, MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

ROLLOUT_CONCURRENCY = 64
MAX_NEW_TOKENS = 2000
MAX_SEQ_LENGTH = 3072  # prompt + response, with margin
LORA_RANK = 64
LORA_ALPHA = 32
MINI_BATCH_SIZES_TO_TRY = [64, 32, 16, 8, 4, 2, 1]


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=45 * 60,
)
def unsloth_smoke_test() -> dict:
    import os

    os.environ["HF_HOME"] = MODEL_CACHE_DIR  # redirect HF cache to the persistent Volume

    import torch

    result = {"steps": []}

    assert torch.cuda.is_available(), "CUDA not available inside the container"
    result["gpu_name"] = torch.cuda.get_device_name(0)
    result["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    result["steps"].append(f"GPU: {result['gpu_name']}, total VRAM: {result['total_vram_gb']} GB")

    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    result["steps"].append("Model + processor loaded (Unsloth FastVisionModel)")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )
    result["steps"].append(f"LoRA applied (r={LORA_RANK}, alpha={LORA_ALPHA}, vision excluded)")

    # Independent verification, same discipline as
    # src/controls/manipulation_check.py - don't just trust the flag,
    # check the actual trainable parameters.
    vision_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    result["vision_trainable_param_count"] = len(vision_trainable)
    result["steps"].append(
        f"Vision-exclusion check: {len(vision_trainable)} trainable vision params (must be 0)"
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    setup_peak = torch.cuda.max_memory_allocated() / 1e9
    result["after_model_load_gb"] = round(setup_peak, 2)
    result["steps"].append(f"After model+LoRA+optimizer setup: {setup_peak:.2f} GB allocated")

    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt

    ex = load_gsm8k("test")[0]
    prompt_text = build_text_prompt(ex.question)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
    pad_id = processor.tokenizer.pad_token_id

    FastVisionModel.for_inference(model)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=True, temperature=0.8)
    torch.cuda.synchronize()
    result["steps"].append("Warm-up generation done (excluded from timing)")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
            temperature=0.8, num_return_sequences=ROLLOUT_CONCURRENCY,
        )
    torch.cuda.synchronize()
    rollout_seconds = time.time() - t0
    gen_peak = torch.cuda.max_memory_allocated() / 1e9
    prompt_len = inputs["input_ids"].shape[1]
    completion_len = gen_out.shape[1] - prompt_len
    total_new_tokens = completion_len * ROLLOUT_CONCURRENCY
    result["rollout_seconds"] = round(rollout_seconds, 2)
    result["rollout_peak_gb"] = round(gen_peak, 2)
    result["rollout_tokens_per_sec"] = round(total_new_tokens / rollout_seconds, 1)
    result["completion_len_reached"] = int(completion_len)
    result["steps"].append(
        f"Rollout: {ROLLOUT_CONCURRENCY} sequences, {completion_len} tokens each, "
        f"{rollout_seconds:.2f}s, {result['rollout_tokens_per_sec']} tok/s, peak {gen_peak:.2f} GB"
    )

    # ---- Training bisection, same methodology as vram_smoke_test_2000cap.py ----
    FastVisionModel.for_training(model)
    torch.cuda.empty_cache()
    attention_mask = (gen_out != pad_id).long()

    training_results = []
    largest_working_mini_batch = None
    for mb_size in MINI_BATCH_SIZES_TO_TRY:
        if mb_size > gen_out.shape[0]:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch_ids = gen_out[:mb_size]
        batch_mask = attention_mask[:mb_size]
        outputs = None
        loss = None
        try:
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_seconds = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            training_results.append(
                {
                    "mini_batch_size": mb_size,
                    "status": "PASS",
                    "peak_gb": round(peak, 2),
                    "step_seconds": round(step_seconds, 2),
                }
            )
            result["steps"].append(
                f"Training mini-batch={mb_size}: PASS, {step_seconds:.2f}s, "
                f"peak={peak:.2f} GB, loss={loss.item():.4f}"
            )
            largest_working_mini_batch = mb_size
            break
        except torch.cuda.OutOfMemoryError as e:
            peak = torch.cuda.max_memory_allocated() / 1e9
            training_results.append(
                {"mini_batch_size": mb_size, "status": "OOM", "peak_gb": round(peak, 2)}
            )
            result["steps"].append(f"Training mini-batch={mb_size}: OOM (peak ~{peak:.2f} GB)")
        finally:
            del outputs, loss
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    result["training_results"] = training_results
    result["largest_working_mini_batch"] = largest_working_mini_batch
    result["status"] = "PASS" if largest_working_mini_batch else "FAIL"
    return result


@app.local_entrypoint()
def run():
    result = unsloth_smoke_test.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    print(f"\nGPU: {result['gpu_name']} ({result['total_vram_gb']} GB total)")
    print(f"Vision-exclusion check: {result['vision_trainable_param_count']} trainable vision params")
    print(
        f"\nRollout: {result['rollout_seconds']}s, {result['rollout_tokens_per_sec']} tok/s, "
        f"peak {result['rollout_peak_gb']} GB"
    )
    print("\nTraining mini-batch bisection:")
    for r in result["training_results"]:
        extra = f", {r['step_seconds']}s" if "step_seconds" in r else ""
        print(f"  mini_batch={r['mini_batch_size']}: {r['status']} (peak {r['peak_gb']} GB){extra}")
    print(f"\nLargest working training mini-batch: {result['largest_working_mini_batch']}")
    print(f"\nStatus: {result['status']}")
