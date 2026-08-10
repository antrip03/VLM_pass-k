"""
Diagnostic, not part of the production pipeline: prints a small number of
RAW model completions (full text, no extraction/filtering applied) for a
couple of GSM8K problems, so a human can visually check whether a
surprisingly low pass@1 (see run_headroom_check_on_modal.py's result) is
a genuine capability limitation or an answer-extraction/prompt-compliance
artifact - exactly the "extraction spot-check" PLAN.md Section 4 item 5
calls for, done here before trusting the headroom check's numbers.

Usage: modal run scripts/inspect_raw_generations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

N_PROBLEMS = 3
N_SAMPLES_PER_PROBLEM = 5
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8

@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=10 * 60,
)
def inspect() -> dict:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt
    from src.metrics.answer_extraction import extract_model_answer, is_correct

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)

    problems = load_gsm8k("test")[:N_PROBLEMS]
    output = []

    for ex in problems:
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

        problem_record = {
            "question": ex.question,
            "ground_truth": ex.final_answer,
            "completions": [],
        }
        for text in completions:
            extracted = extract_model_answer(text)
            correct = is_correct(text, ex.final_answer)
            problem_record["completions"].append(
                {"text": text, "extracted": extracted, "correct": correct}
            )
        output.append(problem_record)

    return {"problems": output}


@app.local_entrypoint()
def run():
    result = inspect.remote()
    for p in result["problems"]:
        print("=" * 80)
        print(f"QUESTION: {p['question']}")
        print(f"GROUND TRUTH: {p['ground_truth']}")
        for i, c in enumerate(p["completions"]):
            print(f"\n--- completion {i} (extracted={c['extracted']!r}, correct={c['correct']}) ---")
            print(c["text"])
