"""
Cheap, real measurement of evaluation throughput per condition (T/D/E),
so the full Phase 1 eval can be priced from data instead of assumption.

Why this exists: PLAN.md 9.3 prices evaluation at ~7.2hr / ~$26 by
applying the batched rollout throughput measured during training
(864.8 tok/s aggregate, 9.2) to the total eval token count. But the
sampler's image path was, until 2026-08-20, one sample per generate()
call - single-sequence decoding, nowhere near that aggregate figure - so
the real cost of the code as written was plausibly several times the
estimate. Rather than argue about the multiplier, measure it: a few
problems at small n, timed per condition, extrapolates honestly to the
real (problems x n x 2 models) grid.

Also validates the two batching changes made the same day (chunked image
generation, batched multi-prompt solves for condition D) actually work on
the real model and the real GPU, at a cost of minutes rather than hours.

Usage:
    modal run scripts/calibrate_eval_throughput.py
    modal run scripts/calibrate_eval_throughput.py --n 16 --problems 2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=60 * 60,
)
def calibrate(n: int, problems: int, image_chunk: int) -> dict:
    import os
    import time

    os.environ["HF_HOME"] = MODEL_CACHE_DIR

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.data.gsm8k_loader import load_gsm8k
    from src.inference.sampler import CONDITION_SAMPLERS

    print(f"Loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    examples = load_gsm8k("test")[:problems]

    results = {}
    for condition in ["T", "D", "E"]:
        sampler_fn = CONDITION_SAMPLERS[condition]
        kwargs = {}
        if condition in ("D", "E"):
            kwargs["image_chunk"] = image_chunk

        start = time.time()
        total_chars = 0
        n_correct = 0
        for ex in examples:
            samples = sampler_fn(model, processor, ex.question, n=n, **kwargs)
            total_chars += sum(len(s.completion) for s in samples)
            n_correct += sum(1 for s in samples if s.extracted_answer == ex.final_answer)
        elapsed = time.time() - start

        per_problem = elapsed / len(examples)
        results[condition] = {
            "elapsed_s": round(elapsed, 1),
            "seconds_per_problem_at_n": round(per_problem, 1),
            "n": n,
            "problems": problems,
            "approx_tokens": round(total_chars / 4),  # rough chars->tokens
            "approx_tok_per_s": round((total_chars / 4) / elapsed, 1),
            "accuracy": round(n_correct / (n * len(examples)), 3),
        }
        print(f"[{condition}] {results[condition]}", flush=True)
        torch.cuda.empty_cache()

    # Extrapolate to the real Phase 1 grid: 50 problems, n=128, 2 models.
    scale_n = 128 / n
    total_s = sum(r["seconds_per_problem_at_n"] * scale_n for r in results.values()) * 50 * 2
    results["_projection"] = {
        "assumes": "50 problems x n=128 x 3 conditions x 2 models (base + RL)",
        "note": "linear-in-n extrapolation; batching means real cost is likely <= this",
        "projected_hours": round(total_s / 3600, 1),
        "projected_usd_at_2_per_hr": round(total_s / 3600 * 2, 2),
        "image_chunk_used": image_chunk,
    }
    print(f"[projection] {results['_projection']}", flush=True)
    return results


@app.local_entrypoint()
def main(n: int = 16, problems: int = 2, image_chunk: int = 16, gpu: str = ""):
    """
    gpu: override the GPU for this calibration (e.g. "A10G"). Eval is
    inference-only - the training-phase VRAM wall that forced A100 for
    training (the LM head's vocab-size logits tensor, see
    scripts/vram_smoke_test_2000cap*.py) does not apply here, so a cheaper
    card is genuinely worth pricing. Decode is memory-bandwidth-bound
    though, so a cheaper-per-hour card is not automatically cheaper per
    run - measure both and compare cost, not rate.
    """
    import json

    fn = calibrate.with_options(gpu=gpu) if gpu else calibrate
    result = fn.remote(n, problems, image_chunk)
    print(json.dumps(result, indent=2))
