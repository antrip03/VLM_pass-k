"""
Diagnostic, not part of the production pipeline: measures the NATURAL
(uncapped-ish) generation length the model actually uses to solve GSM8K
problems, to check empirically whether the planned 500-token cap
(PLAN.md Section 4) is actually sufficient - rather than guess from
priors about "how long should math problems take."

Generates with a generous cap (1500 tokens) so the model can stop
naturally wherever it would, then reports the exact token count of each
completion (measured directly from the output tensor shape, not
estimated from character count) and how many would have been truncated
by a 500-token cap.

Usage: modal run scripts/check_natural_generation_length.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

N_PROBLEMS = 15
N_SAMPLES_PER_PROBLEM = 4
GENEROUS_CAP = 1500  # deliberately generous, to observe natural stopping
CHECK_CAPS = [500, 750, 1000, 1250]  # candidate caps to evaluate against
TEMPERATURE = 0.8


@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=20 * 60,
)
def check_lengths() -> dict:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    eos_id = processor.tokenizer.eos_token_id

    problems = load_gsm8k("test")[:N_PROBLEMS]
    all_lengths: list[int] = []
    hit_cap_count = 0

    for ex in problems:
        prompt_text = build_text_prompt(ex.question)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=GENEROUS_CAP,
                do_sample=True,
                temperature=TEMPERATURE,
                num_return_sequences=N_SAMPLES_PER_PROBLEM,
            )

        for seq in out:
            generated = seq[prompt_len:]
            # Exact token count via the real tensor, not estimated.
            # Trim trailing pad/eos tokens to get the TRUE content length.
            nonpad = generated[generated != processor.tokenizer.pad_token_id]
            length = len(nonpad)
            all_lengths.append(length)
            if length >= GENEROUS_CAP:
                hit_cap_count += 1

    all_lengths.sort()
    n = len(all_lengths)

    def pct(p: float) -> int:
        idx = min(int(p * n), n - 1)
        return all_lengths[idx]

    cap_analysis = {}
    for cap in CHECK_CAPS:
        exceeding = sum(1 for length in all_lengths if length > cap)
        cap_analysis[cap] = {
            "would_truncate": exceeding,
            "would_truncate_pct": round(100 * exceeding / n, 1),
        }

    return {
        "n_generations": n,
        "min": all_lengths[0],
        "p50_median": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "max": all_lengths[-1],
        "hit_generous_cap_count": hit_cap_count,
        "cap_analysis": cap_analysis,
        "all_lengths": all_lengths,
    }


@app.local_entrypoint()
def run():
    result = check_lengths.remote()
    print(f"n_generations: {result['n_generations']}")
    print(f"min: {result['min']}")
    print(f"p50 (median): {result['p50_median']}")
    print(f"p90: {result['p90']}")
    print(f"p99: {result['p99']}")
    print(f"max: {result['max']}")
    print(f"hit the generous {GENEROUS_CAP}-token cap: {result['hit_generous_cap_count']}")
    print("\ncap_analysis (would this cap have truncated the completion?):")
    for cap, stats in result["cap_analysis"].items():
        print(f"  {cap} tokens: {stats['would_truncate']} would truncate ({stats['would_truncate_pct']}%)")
    print(f"\nall_lengths (sorted): {result['all_lengths']}")
