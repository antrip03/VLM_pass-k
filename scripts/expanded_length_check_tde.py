"""
Expanded, more statistically credible version of
check_natural_generation_length.py. That earlier check (60 generations,
Condition T only, base model only) was flagged as too thin to trust for
a tail-risk/truncation question: n=60 cannot meaningfully estimate a p99
(its own p99 collapsed to equal its max - not a real percentile estimate,
just the top observed value), and it only covered Condition T on the
untrained base model, leaving Conditions D and E (PLAN.md Section 1.7)
completely unmeasured.

This script:
  - Increases sample count materially over the earlier n=60/T-only check
    for a better tail estimate, while staying reasonably quick (A10G, no
    LoRA/training - pure measurement).
  - Covers all three conditions: T (text), D (image -> transcribe-only ->
    feed transcription back as text -> solve), E (image -> solve
    end-to-end in one pass) - see PLAN.md Section 1.7 for what each
    isolates.
  - Uses a generous cap (2200) ABOVE both 1500 and 2000 - the two values
    actually being compared right now - so truncation at either can be
    directly observed rather than assumed.
  - Reports cap analysis at both 1500 and 2000 explicitly, per condition,
    so the "keep 1500 to save compute" question can be answered with
    real data instead of extrapolation from the T-only, n=60 check.

REAL BUG FOUND AND FIXED (2026-08-10, first attempt at this script): using
num_return_sequences=N on an IMAGE-conditioned generate() call OOM'd on a
40GB A10G (well, ~22GB usable) - "Tried to allocate 9.71 GiB" inside the
VISION ENCODER's attention (self.visual -> blk -> attn -> SDPA), not the
language model. Root cause: HF's generate() expands ALL inputs (including
pixel_values) by num_return_sequences before the first forward pass, so
requesting 10 samples from one image means the vision encoder processes
10 replicated copies of that image in ONE combined attention call - a
much steeper cost than text-only replication, which is cheap (confirmed:
Condition T, using the exact same num_return_sequences=10 pattern but
text-only, completed successfully in the same run right before Condition
E crashed). Text-only batching (T, and D's solve step, which is also
pure text - feeding a transcription back) still uses num_return_sequences
freely. Image-conditioned calls (E, D's transcribe step) now loop with
num_return_sequences=1 per call instead - the same single-image pattern
already proven safe in configs/modal_app.py's original smoke_test().

Note: builds minimal, self-contained D/E prompt templates inline (Step 6,
src/inference, is not built yet) - not production sampling-harness code,
just enough to measure real length distributions under the real image
pipeline (src/data/render.py) and the real Condition D two-step flow.

Usage: modal run scripts/expanded_length_check_tde.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

N_PROBLEMS = 10
TEXT_SAMPLES = 10  # Condition T: pure text, num_return_sequences batching is cheap and safe
IMAGE_SAMPLES = 6  # Conditions E and D-transcribe: image-conditioned, looped one-at-a-time
                    # (see module docstring - num_return_sequences on an image input OOM'd)
GENEROUS_CAP = 2200  # above both 1500 and 2000 - the two caps under comparison
TRANSCRIBE_CAP = 300  # transcription should be short; generous for a 1-3 sentence problem
CHECK_CAPS = [1000, 1250, 1500, 1750, 2000]
TEMPERATURE = 0.8

E_INSTRUCTION = (
    "Solve the math problem shown in the image step by step. Then, on its "
    "own final line, write your answer in EXACTLY this format, with "
    "nothing else on that line and nothing after it:\n"
    "#### <number>"
)
D_TRANSCRIBE_INSTRUCTION = (
    "Transcribe the math problem shown in the image exactly as written, "
    "word for word. Do not solve it or explain it - only output the "
    "transcribed problem text."
)


@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=30 * 60,
)
def expanded_length_check() -> dict:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_text_prompt
    from src.data.render import render_problem_image

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    pad_id = processor.tokenizer.pad_token_id

    lengths = {"T": [], "D_transcribe": [], "D_solve": [], "E": []}
    problems = load_gsm8k("test")[:N_PROBLEMS]

    for ex in problems:
        img = render_problem_image(ex.question)

        # --- Condition T: plain text ---
        t_prompt = build_text_prompt(ex.question)
        t_messages = [{"role": "user", "content": [{"type": "text", "text": t_prompt}]}]
        t_text = processor.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True)
        t_inputs = processor(text=[t_text], return_tensors="pt").to("cuda")
        t_plen = t_inputs["input_ids"].shape[1]
        with torch.no_grad():
            t_out = model.generate(
                **t_inputs, max_new_tokens=GENEROUS_CAP, do_sample=True,
                temperature=TEMPERATURE, num_return_sequences=TEXT_SAMPLES,
            )
        for seq in t_out:
            nonpad = seq[t_plen:][seq[t_plen:] != pad_id]
            lengths["T"].append(len(nonpad))

        # --- Condition E: image, solve end-to-end ---
        # One sample per generate() call, NOT num_return_sequences (see module
        # docstring - replicating an image via num_return_sequences OOM'd the
        # vision encoder's attention).
        e_messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": E_INSTRUCTION}]}
        ]
        e_text = processor.apply_chat_template(e_messages, tokenize=False, add_generation_prompt=True)
        e_vision, _ = process_vision_info(e_messages)
        e_inputs = processor(text=[e_text], images=e_vision, return_tensors="pt").to("cuda")
        e_plen = e_inputs["input_ids"].shape[1]
        for _ in range(IMAGE_SAMPLES):
            with torch.no_grad():
                e_out = model.generate(
                    **e_inputs, max_new_tokens=GENEROUS_CAP, do_sample=True, temperature=TEMPERATURE,
                )
            nonpad = e_out[0, e_plen:][e_out[0, e_plen:] != pad_id]
            lengths["E"].append(len(nonpad))

        # --- Condition D, step 1: transcribe only (same one-at-a-time fix) ---
        d1_messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": D_TRANSCRIBE_INSTRUCTION}]}
        ]
        d1_text = processor.apply_chat_template(d1_messages, tokenize=False, add_generation_prompt=True)
        d1_vision, _ = process_vision_info(d1_messages)
        d1_inputs = processor(text=[d1_text], images=d1_vision, return_tensors="pt").to("cuda")
        d1_plen = d1_inputs["input_ids"].shape[1]
        transcriptions = []
        for _ in range(IMAGE_SAMPLES):
            with torch.no_grad():
                d1_out = model.generate(
                    **d1_inputs, max_new_tokens=TRANSCRIBE_CAP, do_sample=True, temperature=TEMPERATURE,
                )
            nonpad = d1_out[0, d1_plen:][d1_out[0, d1_plen:] != pad_id]
            lengths["D_transcribe"].append(len(nonpad))
            transcriptions.append(processor.tokenizer.decode(nonpad, skip_special_tokens=True))

        # --- Condition D, step 2: feed each real transcription back, solve ---
        # Different text per row (real transcriptions differ) -> needs left-padding,
        # not num_return_sequences (which only replicates ONE shared prompt).
        processor.tokenizer.padding_side = "left"
        solve_prompts = [
            processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": build_text_prompt(t)}]}],
                tokenize=False, add_generation_prompt=True,
            )
            for t in transcriptions
        ]
        d2_inputs = processor(text=solve_prompts, return_tensors="pt", padding=True).to("cuda")
        d2_plen = d2_inputs["input_ids"].shape[1]
        with torch.no_grad():
            d2_out = model.generate(**d2_inputs, max_new_tokens=GENEROUS_CAP, do_sample=True, temperature=TEMPERATURE)
        processor.tokenizer.padding_side = "right"  # restore default
        for seq in d2_out:
            nonpad = seq[d2_plen:][seq[d2_plen:] != pad_id]
            lengths["D_solve"].append(len(nonpad))

    def summarize(vals: list[int]) -> dict:
        vals = sorted(vals)
        n = len(vals)

        def pct(p):
            return vals[min(int(p * n), n - 1)]

        return {
            "n": n,
            "min": vals[0],
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p99": pct(0.99),
            "max": vals[-1],
            "hit_generous_cap": sum(1 for v in vals if v >= GENEROUS_CAP),
            "cap_analysis": {
                cap: {
                    "would_truncate": sum(1 for v in vals if v > cap),
                    "pct": round(100 * sum(1 for v in vals if v > cap) / n, 1),
                }
                for cap in CHECK_CAPS
            },
        }

    return {
        "T": summarize(lengths["T"]),
        "D_transcribe": summarize(lengths["D_transcribe"]),
        "D_solve": summarize(lengths["D_solve"]),
        "E": summarize(lengths["E"]),
    }


@app.local_entrypoint()
def run():
    result = expanded_length_check.remote()
    for cond in ["T", "D_transcribe", "D_solve", "E"]:
        s = result[cond]
        print(f"\n=== Condition {cond} (n={s['n']}) ===")
        print(f"  min={s['min']} p50={s['p50']} p90={s['p90']} p99={s['p99']} max={s['max']}")
        print(f"  hit generous cap ({2200}): {s['hit_generous_cap']}")
        for cap, stats in s["cap_analysis"].items():
            print(f"  cap={cap}: {stats['would_truncate']} would truncate ({stats['pct']}%)")
