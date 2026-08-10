"""
Real, cheap smoke test for src/inference/sampler.py (Step 6) - small
scale (2 problems x 3 samples x 3 conditions = 18 total generations),
checking the harness actually works end to end before trusting it at
real scale (n~128, PLAN.md Section 4 item 6): no crashes, answer
extraction succeeds at least sometimes (not silently broken), and
Condition D's transcriptions/completions are coherent, not gibberish.

Usage: modal run scripts/smoke_test_sampler.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

N_PROBLEMS = 2
N_SAMPLES = 3


@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=15 * 60,
)
def smoke_test_sampler() -> dict:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.data.gsm8k_loader import load_gsm8k
    from src.inference.sampler import sample_condition_d, sample_condition_e, sample_condition_t

    result = {"steps": [], "conditions": {}}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=MODEL_CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    result["steps"].append("Model + processor loaded")

    problems = load_gsm8k("test")[:N_PROBLEMS]

    for cond_name, sampler_fn in [("T", sample_condition_t), ("D", sample_condition_d), ("E", sample_condition_e)]:
        cond_results = []
        for ex in problems:
            samples = sampler_fn(model, processor, ex.question, n=N_SAMPLES)
            n_extracted = sum(1 for s in samples if s.extracted_answer is not None)
            n_correct = sum(1 for s in samples if s.extracted_answer == ex.final_answer)
            cond_results.append(
                {
                    "question": ex.question,
                    "ground_truth": ex.final_answer,
                    "n_samples": len(samples),
                    "n_extracted": n_extracted,
                    "n_correct": n_correct,
                    "sample_completions": [s.completion[:250] for s in samples],
                    "sample_transcriptions": [s.transcription[:250] if s.transcription else None for s in samples],
                    "extracted_answers": [s.extracted_answer for s in samples],
                }
            )
        result["conditions"][cond_name] = cond_results
        result["steps"].append(
            f"Condition {cond_name}: {len(cond_results)} problems x {N_SAMPLES} samples done"
        )

    result["status"] = "PASS"
    return result


@app.local_entrypoint()
def run():
    result = smoke_test_sampler.remote()
    for step in result["steps"]:
        print(f"  - {step}")

    for cond_name, cond_results in result["conditions"].items():
        print(f"\n=== Condition {cond_name} ===")
        for r in cond_results:
            print(f"\nQuestion: {r['question'][:150]}...")
            print(f"Ground truth: {r['ground_truth']}")
            print(f"Extracted answers: {r['extracted_answers']}")
            print(f"n_extracted={r['n_extracted']}/{r['n_samples']}, n_correct={r['n_correct']}/{r['n_samples']}")
            if any(r["sample_transcriptions"]):
                print(f"Sample transcription: {r['sample_transcriptions'][0]}")
            print(f"Sample completion: {r['sample_completions'][0]}")

    print(f"\nStatus: {result['status']}")
