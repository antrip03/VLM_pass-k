"""
Compliance-matching causal control (mentor review, 2026-08-25, item 8 -
"the experiment that would prove the mechanism instead of inferring it").

WHAT THIS TESTS
----------------
Every result so far is OBSERVATIONAL: compliance and the strict Delta
move together across training, across datasets, across conditions. That
is strong correlational evidence, but a reviewer can still ask "did you
actually manipulate anything, or just watch two things co-vary?"

This experiment manipulates the mediator directly, with NO training:
run the untrained base model on condition E (the condition with the
largest compliance gap: 79.0% base vs 93.5% RL) with a prompt that adds
ONE concrete example of the "#### N" marker line - no worked reasoning,
no extra content, nothing else changed. See src/data/prompts.py's
E_CONDITION_INSTRUCTION_FEWSHOT docstring for why a format exemplar was
used instead of a full worked-example few-shot (the latter would also
prime reasoning style, confounding the manipulation).

THE PREDICTION
---------------
If format compliance is really what drove the strict Delta:
  - The exemplar should raise base-model compliance on E, without
    training and without a training reward.
  - Comparing (few-shot base) vs (the ALREADY-EVALUATED, zero-shot RL
    model) under STRICT scoring, the Delta should shrink toward the
    FORMAT-AGNOSTIC Delta (~0), because the compliance gap that inflated
    it has now been closed by prompting alone rather than by training.
  - Format-agnostic accuracy should be UNCHANGED by the exemplar - it is
    not supposed to teach reasoning, only reveal it.

If instead few-shot compliance does NOT rise, or strict Delta does NOT
move, the mechanism as described in FINDINGS.md section 5.4 is wrong and
that section needs revising, not defending.

SCOPE, DELIBERATELY MINIMAL
----------------------------
Base model only (no RL re-run - its zero-shot E records already exist in
eval_results/records_rl.fallback.parquet). Condition E only (D/T are the
same mechanism at smaller effect size; E is where it is largest and
where the paper's contribution lives). n=64 (not 128): the quantity of
interest is pass@1 and the compliance rate, both dominated by
between-problem variance, not within-problem sample count - halving n
roughly halves cost for negligible CI cost.

    modal run --detach scripts/run_compliance_matching_control.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

RESULTS_DIR = "/results"
results_volume = modal.Volume.from_name("grpo-vlm-phase1b-results", create_if_missing=True)

EVAL_GPU = "A10G"
N_SAMPLES = 64
N_PROBLEMS = 50
CONDITION = "E"


@app.function(
    image=image,
    gpu=EVAL_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, RESULTS_DIR: results_volume},
    timeout=4 * 60 * 60,
)
def eval_base_fewshot() -> dict:
    """Base model, condition E, format-exemplar prompt, no training."""
    import json
    import os
    import time

    os.environ["HF_HOME"] = MODEL_CACHE_DIR

    from src.data.gsm8k_loader import load_gsm8k
    from src.data.prompts import build_image_prompt_fewshot
    from src.inference.checkpoint_loading import load_model_at_step
    from src.inference.run_sampling import SamplingRecord, save_sampling_records
    from src.inference.sampler import sample_condition_e
    from src.metrics.pass_at_k import mean_pass_at_k

    print(f"=== compliance-matching control: base/E, format-exemplar prompt, "
          f"n={N_SAMPLES}, problems={N_PROBLEMS} ===", flush=True)
    t0 = time.time()

    model, processor, vision_sha = load_model_at_step(
        model_id=MODEL_ID, cache_dir=MODEL_CACHE_DIR, step=None,
    )
    print(f"[base] vision-tower sha256: {vision_sha}", flush=True)

    instruction = build_image_prompt_fewshot()
    examples = load_gsm8k("test")[:N_PROBLEMS]

    records: list[SamplingRecord] = []
    started = time.time()
    for i, problem in enumerate(examples, start=1):
        samples = sample_condition_e(
            model, processor, problem.question, n=N_SAMPLES, instruction_override=instruction,
        )
        for sample_idx, sample in enumerate(samples):
            from src.metrics.answer_extraction_fallback import is_correct_with_fallback

            fb_correct, fb_method = is_correct_with_fallback(sample.completion, problem.final_answer)
            records.append(
                SamplingRecord(
                    problem_idx=problem.idx,
                    question=problem.question,
                    ground_truth=problem.final_answer,
                    condition=CONDITION,
                    sample_idx=sample_idx,
                    completion=sample.completion,
                    extracted_answer=sample.extracted_answer,
                    transcription=None,
                    correct=(sample.extracted_answer == problem.final_answer),
                    correct_fallback=fb_correct,
                    extraction_method=fb_method,
                )
            )
        elapsed = time.time() - started
        eta = (N_PROBLEMS - i) * (elapsed / i)
        n_so_far = len(records)
        strict_acc = sum(1 for r in records if r.correct) / n_so_far
        fair_acc = sum(1 for r in records if r.correct_fallback) / n_so_far
        print(f"[eval] fewshot/base problem {i}/{N_PROBLEMS} | elapsed {elapsed/60:.1f}min "
              f"| ETA {eta/60:.1f}min | acc strict {strict_acc:.3f} / fallback {fair_acc:.3f}", flush=True)

    out_path = f"{RESULTS_DIR}/compliance_control_records_base_fewshot.parquet"
    save_sampling_records(records, out_path)
    results_volume.commit()

    by_problem: dict[int, list[bool]] = {}
    by_problem_fb: dict[int, list[bool]] = {}
    for r in records:
        by_problem.setdefault(r.problem_idx, []).append(r.correct)
        by_problem_fb.setdefault(r.problem_idx, []).append(r.correct_fallback)
    pp = [(len(v), sum(v)) for _, v in sorted(by_problem.items())]
    pp_fb = [(len(v), sum(v)) for _, v in sorted(by_problem_fb.items())]

    n_unparsed = sum(1 for r in records if r.extracted_answer is None)
    summary = {
        "condition": CONDITION,
        "prompt_variant": "fewshot_format_exemplar",
        "n": N_SAMPLES,
        "problems": len(examples),
        "records": len(records),
        "elapsed_hr": round((time.time() - t0) / 3600, 2),
        "vision_sha256": vision_sha,
        "format_compliance_rate": round(1 - n_unparsed / len(records), 4),
        "pass_at_1_strict": round(mean_pass_at_k(pp, 1), 4),
        "pass_at_1_fallback": round(mean_pass_at_k(pp_fb, 1), 4),
        "out_path": out_path,
    }
    with open(f"{RESULTS_DIR}/compliance_control_summary_base_fewshot.json", "w") as f:
        json.dump(summary, f, indent=2)
    results_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main():
    print(f"Compliance-matching control: base model, condition E, format-exemplar prompt, "
          f"n={N_SAMPLES}, {N_PROBLEMS} problems.")
    print("Waiting for result (use `modal run --detach` for safety against dropped connections)...")
    result = eval_base_fewshot.remote()
    print(f"\nDone: {result}")
    print("\nNext: python3 scripts/analyze_compliance_control.py")
