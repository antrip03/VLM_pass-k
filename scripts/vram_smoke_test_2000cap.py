"""
Real VRAM smoke test on A100 40GB (PLAN.md Step 5): does a 2000-token
response cap fit in memory for both the generation (rollout) phase and
the training (forward+backward+optimizer) phase, on this exact model
with the exact LoRA config we intend to train with?

Two genuinely different memory regimes are tested, not just one, because
they stress different things:

  1. GENERATION at real concurrency (64 concurrent sequences - one real
     GSM8K prompt, replicated via num_return_sequences=64). Memory here
     only depends on batch size and sequence length, not content
     diversity, so replicating one prompt is a faithful and much simpler
     proxy than sampling 64 different prompts. This stresses the KV
     cache, which grows with concurrent sequences x sequence length.

  2. TRAINING (forward+backward+optimizer.step on the generated
     sequences). This stresses something KV cache does NOT: the language
     model head projects every token position to a 151,936-token
     vocabulary. Naively computing loss with HF's `labels=` argument
     materializes a full [batch, seq_len, vocab_size] logits tensor
     before applying cross-entropy. For a real ~2000+-token batch of 64
     sequences this is enormous even in bf16:
       64 x ~2200 x 151936 x 2 bytes =~ 43 GB for the logits tensor ALONE
     - before gradients, weights, or KV cache. This is NOT a bug to route
     around by shrinking the cap; it's exactly why every real GRPO/PPO
     implementation (including the real veRL example this project is
     following, examples/tuning/lora/run_qwen2_5_vl_7b_fsdp.sh) trains in
     MINI-BATCHES far smaller than the full rollout batch
     (train_batch_size=512, ppo_mini_batch_size=128 there) - not only for
     gradient-accumulation smoothing but to keep this logits tensor
     bounded. So this test does not try one single 64-wide backward pass
     and call it a verdict - it bisects mini-batch size
     (64, 32, 16, 8, 4, 2, 1) and reports the largest that actually fits,
     which is a directly useful number for configuring
     ppo_mini_batch_size in the real training run.

HONEST LIMITATION: this uses plain HF `model(...).loss` (full teacher-
forcing cross-entropy), which does NOT use a fused/chunked cross-entropy
kernel (e.g. Liger Kernel) that avoids ever materializing the full
logits tensor. If veRL's actual training path uses such a kernel, its
real ceiling could be HIGHER than what this test finds - so the
mini-batch numbers below are a conservative lower bound on what will
work, not a hard ceiling on veRL itself. What this test DOES answer
directly and reliably, with no such caveat: whether 2000-token
GENERATION at real concurrency fits (no vocab-logits issue there).

Usage: modal run scripts/vram_smoke_test_2000cap.py
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import FULL_RUN_GPU, MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

GENERATION_CONCURRENCY = 64  # one real GSM8K prompt, replicated via num_return_sequences
MAX_NEW_TOKENS = 2000
LORA_RANK = 64
LORA_ALPHA = 32
MINI_BATCH_SIZES_TO_TRY = [64, 32, 16, 8, 4, 2, 1]  # largest first, bisect down


@app.function(
    image=image,
    gpu=FULL_RUN_GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=45 * 60,
)
def vram_smoke_test() -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.controls.manipulation_check import build_lora_target_modules
    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt

    result = {"steps": []}

    assert torch.cuda.is_available(), "CUDA not available inside the container"
    result["gpu_name"] = torch.cuda.get_device_name(0)
    result["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    result["steps"].append(f"GPU: {result['gpu_name']}, total VRAM: {result['total_vram_gb']} GB")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    pad_id = processor.tokenizer.pad_token_id
    result["steps"].append("Model + processor loaded")

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
    result["steps"].append(
        f"LoRA applied (r={LORA_RANK}, alpha={LORA_ALPHA}, {len(target_modules)} target modules)"
    )

    setup_peak = torch.cuda.max_memory_allocated() / 1e9
    result["after_model_load_gb"] = round(setup_peak, 2)
    result["steps"].append(f"After model+LoRA+optimizer setup: {setup_peak:.2f} GB allocated")

    # ---- Phase 1: generation (rollout) at real concurrency ----
    ex = load_gsm8k("test")[0]
    prompt_text = build_text_prompt(ex.question)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")

    torch.cuda.reset_peak_memory_stats()
    model.eval()
    gen_out = None
    try:
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=GENERATION_CONCURRENCY,
            )
        gen_peak = torch.cuda.max_memory_allocated() / 1e9
        result["generation_status"] = "PASS"
        result["generation_peak_gb"] = round(gen_peak, 2)
        result["generation_seq_len"] = int(gen_out.shape[1])
        result["steps"].append(
            f"Generation PASS: {GENERATION_CONCURRENCY} concurrent sequences, "
            f"max_new_tokens={MAX_NEW_TOKENS}, actual total seq_len={gen_out.shape[1]}, "
            f"peak allocated={gen_peak:.2f} GB"
        )
    except torch.cuda.OutOfMemoryError as e:
        result["generation_status"] = "OOM"
        result["generation_error"] = str(e)[:500]
        result["steps"].append(f"Generation OOM at concurrency={GENERATION_CONCURRENCY}: {e}")

    torch.cuda.empty_cache()

    # ---- Phase 2: training (forward+backward+optimizer), bisecting mini-batch size ----
    training_results = []
    largest_working_mini_batch = None
    if gen_out is not None:
        model.train()
        attention_mask = (gen_out != pad_id).long()
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
                outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                peak = torch.cuda.max_memory_allocated() / 1e9
                training_results.append(
                    {"mini_batch_size": mb_size, "status": "PASS", "peak_gb": round(peak, 2)}
                )
                result["steps"].append(
                    f"Training mini-batch={mb_size}: PASS, peak={peak:.2f} GB, loss={loss.item():.4f}"
                )
                largest_working_mini_batch = mb_size
                break  # found the largest that fits, no need to test smaller
            except torch.cuda.OutOfMemoryError as e:
                peak = torch.cuda.max_memory_allocated() / 1e9
                training_results.append(
                    {"mini_batch_size": mb_size, "status": "OOM", "peak_gb": round(peak, 2)}
                )
                result["steps"].append(
                    f"Training mini-batch={mb_size}: OOM (peak before failure ~{peak:.2f} GB)"
                )
            finally:
                del outputs, loss
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

    result["training_results"] = training_results
    result["largest_working_mini_batch"] = largest_working_mini_batch
    result["status"] = (
        "PASS" if result.get("generation_status") == "PASS" and largest_working_mini_batch else "FAIL"
    )
    return result


@app.local_entrypoint()
def run():
    result = vram_smoke_test.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    print(f"\nGPU: {result['gpu_name']} ({result['total_vram_gb']} GB total)")
    print(f"Generation status: {result.get('generation_status')}")
    if result.get("generation_status") == "PASS":
        print(f"  peak during generation: {result['generation_peak_gb']} GB")
        print(f"  actual sequence length reached: {result['generation_seq_len']}")
    print("\nTraining mini-batch bisection:")
    for r in result.get("training_results", []):
        print(f"  mini_batch={r['mini_batch_size']}: {r['status']} (peak {r['peak_gb']} GB)")
    print(f"\nLargest working training mini-batch: {result.get('largest_working_mini_batch')}")
    print(f"\nOverall status: {result['status']}")
