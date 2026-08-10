"""
Runs the real headroom pre-check (PLAN.md Phase 0 item 2) against
Qwen2.5-VL-3B-Instruct on live Modal infrastructure: samples n=64
completions per problem (n >> k_high=16, per the n/k ratio lesson learned
while sanity-checking check_headroom locally - see
scripts/sanity_check_metrics.py) across 25 real GSM8K text problems, then
reports pass@1, pass@16, and the go/no-go verdict.

Usage: modal run scripts/run_headroom_check_on_modal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

N_PROBLEMS = 25
N_SAMPLES_PER_PROBLEM = 64  # >> k_high=16
K_HIGH = 16
MAX_NEW_TOKENS = 500  # matches PLAN.md's standardized generation cap
TEMPERATURE = 0.8



@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=30 * 60,
)
def run_headroom_check() -> dict:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.controls.headroom_check import check_headroom
    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt
    from src.metrics.answer_extraction import is_correct

    result = {"steps": []}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    result["steps"].append("Model loaded")

    problems = load_gsm8k("test")[:N_PROBLEMS]
    result["steps"].append(f"Loaded {len(problems)} GSM8K test problems")

    results_per_problem: list[tuple[int, int]] = []
    for idx, ex in enumerate(problems):
        prompt_text = build_text_prompt(ex.question)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                num_return_sequences=N_SAMPLES_PER_PROBLEM,
            )
        completions = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        c = sum(1 for text in completions if is_correct(text, ex.final_answer))
        results_per_problem.append((N_SAMPLES_PER_PROBLEM, c))
        result["steps"].append(
            f"Problem {idx} (idx={ex.idx}): {c}/{N_SAMPLES_PER_PROBLEM} correct"
        )

    report = check_headroom(results_per_problem, k_high=K_HIGH)
    result["headroom_report"] = report
    result["status"] = "PASS" if report["has_headroom"] else "NO_HEADROOM"
    return result


@app.local_entrypoint()
def run():
    result = run_headroom_check.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    print(f"\nheadroom_report: {result['headroom_report']}")
    print(f"\nStatus: {result['status']}")
