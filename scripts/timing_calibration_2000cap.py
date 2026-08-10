"""
Real wall-clock timing calibration at the 2000-token response cap (PLAN.md
Section 3, Phase 0 item 6: "measure real tokens/sec on the actual hardware
being used - do not trust the estimates in Section 9 blindly").

Builds on two already-validated findings from vram_smoke_test_2000cap.py
(run earlier, same session):
  - Generation at 64 concurrent sequences, 2000-token cap: fits comfortably
    (peak 10.98 GB / 42.41 GB).
  - Training (forward+backward+optimizer) at this response length: only a
    micro-batch of 4 fits on a single 40GB A100 (peak 29.32 GB); larger
    micro-batches (8/16/32/64) OOM due to the LM head's
    [batch, seq_len, vocab_size] logits tensor, not KV cache.

This script does NOT re-run that memory bisection (already answered) - it
measures real WALL-CLOCK TIME for exactly those two already-confirmed-
working configurations, so the compute estimates in PLAN.md Section 9
(written before any real Modal profiling existed, at a 500-token cap, not
2000) can be replaced with real numbers rather than trusted blindly.

Excludes a warm-up call from each timed measurement (first-call CUDA
kernel compilation / cudnn autotuning overhead is not representative of
steady-state per-step cost across hundreds of training steps).

Usage: modal run scripts/timing_calibration_2000cap.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import FULL_RUN_GPU, MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

ROLLOUT_CONCURRENCY = 64  # matches vram_smoke_test_2000cap.py's confirmed-safe generation size
MAX_NEW_TOKENS = 2000
LORA_RANK = 64
LORA_ALPHA = 32
TRAINING_MICRO_BATCH = 4  # matches vram_smoke_test_2000cap.py's confirmed largest working size


@app.function(
    image=image,
    gpu=FULL_RUN_GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=30 * 60,
)
def timing_calibration() -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.controls.manipulation_check import build_lora_target_modules
    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt

    result = {"steps": []}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    pad_id = processor.tokenizer.pad_token_id

    target_modules = build_lora_target_modules(model)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    result["steps"].append(f"Model + LoRA (r={LORA_RANK}) ready")

    ex = load_gsm8k("test")[0]
    prompt_text = build_text_prompt(ex.question)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=True, temperature=0.8)
    torch.cuda.synchronize()
    result["steps"].append("Warm-up generation done (excluded from timing)")

    # ---- Timed rollout: 64 concurrent sequences, 2000-token cap ----
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=ROLLOUT_CONCURRENCY,
        )
    torch.cuda.synchronize()
    rollout_seconds = time.time() - t0
    prompt_len = inputs["input_ids"].shape[1]
    completion_len = gen_out.shape[1] - prompt_len
    total_new_tokens = completion_len * ROLLOUT_CONCURRENCY
    result["rollout_seconds"] = round(rollout_seconds, 2)
    result["rollout_concurrency"] = ROLLOUT_CONCURRENCY
    result["completion_len_reached"] = int(completion_len)
    result["rollout_tokens_per_sec"] = round(total_new_tokens / rollout_seconds, 1)
    result["steps"].append(
        f"Rollout: {ROLLOUT_CONCURRENCY} sequences, reached {completion_len} tokens each, "
        f"{rollout_seconds:.2f}s wall-clock, {result['rollout_tokens_per_sec']} tok/s aggregate"
    )

    # ---- Timed training micro-batch step (forward+backward+optimizer) ----
    model.train()
    attention_mask = (gen_out != pad_id).long()
    batch_ids = gen_out[:TRAINING_MICRO_BATCH]
    batch_mask = attention_mask[:TRAINING_MICRO_BATCH]

    optimizer.zero_grad()
    warm_out = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
    warm_out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    result["steps"].append("Warm-up training step done (excluded from timing)")

    torch.cuda.synchronize()
    t0 = time.time()
    outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    train_step_seconds = time.time() - t0
    result["train_micro_batch_size"] = TRAINING_MICRO_BATCH
    result["train_step_seconds"] = round(train_step_seconds, 2)
    result["train_seq_len"] = int(batch_ids.shape[1])
    result["steps"].append(
        f"Training micro-batch={TRAINING_MICRO_BATCH}, seq_len={batch_ids.shape[1]}: "
        f"{train_step_seconds:.2f}s wall-clock (loss={loss.item():.4f})"
    )

    result["status"] = "PASS"
    return result


@app.local_entrypoint()
def run():
    result = timing_calibration.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    print(
        f"\nRollout: {result['rollout_seconds']}s for {result['rollout_concurrency']} sequences "
        f"({result['completion_len_reached']} tokens each) = {result['rollout_tokens_per_sec']} tok/s"
    )
    print(
        f"Training step: {result['train_step_seconds']}s for micro-batch="
        f"{result['train_micro_batch_size']} at seq_len={result['train_seq_len']}"
    )
    print(f"\nStatus: {result['status']}")
