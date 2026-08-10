"""
Custom veRL reward function (PLAN.md Section 3 Step 5), wiring this
project's own answer-extraction logic (src/metrics/answer_extraction.py)
into training instead of veRL's built-in gsm8k reward.

Why not just use veRL's built-in reward: verified against the real,
current veRL source (verl/utils/reward_score/gsm8k.py, fetched verbatim
2026-08-10) that its default `method="strict"` only matches "#### N" -
the exact same gap this project already found and fixed in Step 4 (a
live spot-check proved Qwen2.5-VL-3B-Instruct frequently answers
correctly via "\\boxed{N}" instead of "####", and a strict-only
extractor silently scored those as wrong). veRL does ship a "flexible"
fallback mode, but it grabs the last bare number in the text regardless
of any structural marker - a blunter, higher-risk mechanism than this
project's own \\boxed{}-aware fallback chain. Using the SAME extraction
function for both training reward and evaluation scoring also keeps the
two consistent, rather than training against one definition of "correct"
and evaluating against another.

Signature confirmed against two real, independent sources (not guessed):
verl's own custom-reward-function docs and a complete, working
Modal+verl+GRPO example (modal.com/docs/examples/grpo_verl), both fetched
verbatim 2026-08-10:
    def compute_score(data_source: str, solution_str: str,
                       ground_truth: str, extra_info: dict) -> float

Scoring is a plain binary 1.0/0.0 (correct/not), matching both real
references above - neither uses partial "format credit" for a
correctly-formatted-but-wrong answer, and PLAN.md never called for one,
so this doesn't invent an unjustified extra hyperparameter.

Usage: veRL config sets
    custom_reward_function.path=<path to this file>
    custom_reward_function.name=compute_score   (the default name, but
        set explicitly in the training config for clarity)
"""

from __future__ import annotations

from src.metrics.answer_extraction import is_correct

CORRECT_SCORE = 1.0
INCORRECT_SCORE = 0.0


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict
) -> float:
    """
    data_source, extra_info: unused here (this project trains on a
    single dataset, GSM8K, with no per-source or per-example scoring
    variation) - accepted anyway because veRL's reward-loop calls every
    custom_reward_function with this exact four-argument signature
    regardless of whether a given implementation needs all of them.
    """
    return CORRECT_SCORE if is_correct(solution_str, ground_truth) else INCORRECT_SCORE
