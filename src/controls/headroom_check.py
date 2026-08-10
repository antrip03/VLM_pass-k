"""
Headroom pre-check (PLAN.md Phase 0 item 2; mandatory, non-negotiable
before any training compute on Qwen3-VL-2B-Instruct per Phase 2B):
confirms a base model has enough "latent capability" for GRPO to
plausibly sharpen, before spending any training compute on it.

The diagnostic: sample pass@1 and pass@16 on a small set of TEXT problems
using the untrained base model. If pass@16 is only marginally higher than
pass@1, the model rarely/never reaches the correct answer even given many
tries - there's no latent capability for RL to amplify, and training
would likely produce a small/noisy Delta_text that isn't a design flaw,
just a ceiling/floor effect on this specific model (exactly the concern
flagged for Qwen3-VL-2B-Instruct given its distillation-acquired
reasoning - catch it here, cheaply, rather than discover it after a full
training run).

VERIFICATION STATUS: run for real against Qwen2.5-VL-3B-Instruct on live
Modal infrastructure (2026-08-10, scripts/run_headroom_check_on_modal.py),
25 GSM8K text problems, n=64 samples/problem. Result: pass@1=0.513,
pass@16=0.912, gap=0.399 - clear headroom, has_headroom=True.

That result came from a SECOND run, not the first. The first run reported
an implausible pass@1=0.0175 (1.75%) - suspiciously low for a 3B instruct
model on GSM8K. Rather than accept it, a spot-check of raw completions
(scripts/inspect_raw_generations.py) found the real cause: the model
frequently reasoned correctly but wrote "\\boxed{18}" instead of the
requested "#### 18", which a "#### only" extractor completely missed -
genuinely correct completions were being scored as wrong. Fixed in
src/metrics/answer_extraction.py (added a \\boxed{} fallback) and
src/data/prompts.py (a more forceful format instruction); re-running with
both fixes produced the 0.513 pass@1 reported above. Left as a case study
in this module because it's the clearest illustration in this codebase so
far of why PLAN.md's calibration/spot-check steps aren't optional
paperwork - an implausible number is a signal to investigate, not a
result to report.
"""

from __future__ import annotations

from src.metrics.pass_at_k import mean_pass_at_k

DEFAULT_MIN_GAP = 0.05  # pass@k_high - pass@1 must exceed this to "pass"


def check_headroom(
    results_per_problem: list[tuple[int, int]],
    k_high: int = 16,
    min_gap: float = DEFAULT_MIN_GAP,
) -> dict:
    """
    results_per_problem: list of (n, c) pairs, one per problem, from
    sampling the untrained base model on a small text-problem set (n
    samples drawn, c of them correct).

    Returns pass@1, pass@k_high, their gap, and a go/no-go verdict.

    This is a coarse, cheap gate (fixed threshold, no confidence
    interval) - intentionally not the rigorous statistical treatment
    Step 7 applies to the main evaluation. It exists to catch an
    obviously-too-weak model cheaply before spending training compute,
    not to be the final scientific word on headroom.
    """
    if not results_per_problem:
        raise ValueError("results_per_problem must not be empty")

    p1 = mean_pass_at_k(results_per_problem, k=1)
    p_high = mean_pass_at_k(results_per_problem, k=k_high)
    gap = p_high - p1

    return {
        "pass_at_1": p1,
        f"pass_at_{k_high}": p_high,
        "gap": gap,
        "has_headroom": gap >= min_gap,
        "min_gap_threshold": min_gap,
        "n_problems": len(results_per_problem),
    }
