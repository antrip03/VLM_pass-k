"""
RANDOMISED reward function - the training signal for the Phase 1b Step 4
control.

WHAT THIS CONTROL IS FOR
------------------------
Shao et al., "Spurious Rewards: Rethinking Training Signals in RLVR"
(arXiv 2506.10947), show that on Qwen models RLVR produces large gains
even when the reward carries no information about correctness: on
Qwen2.5-Math-7B, MATH-500 improves +21.4% with a RANDOM reward against
+29.1% with the ground-truth reward. The effect is Qwen-specific - it
"generally fails" on Llama3 or OLMo2 - and their explicit warning is that
"RLVR research validated solely on Qwen models might not generalize".

This project trains Qwen2.5-VL-3B. So a reviewer who knows that paper
will ask whether Delta_text = +0.0587 reflects learning from the
correctness signal at all, and "different benchmark, different model" is
not an answer. The direct answer is to run the same training with the
reward severed from correctness and show the gain does not survive.

WHY BERNOULLI(0.5) AND NOT SOMETHING ELSE
-----------------------------------------
gamma = 0.5 matches Shao et al.'s random-reward condition. It also keeps
GRPO's group-relative advantage alive: with group_size=8, a Bernoulli(0.5)
draw gives a mixed group almost always (all-same happens with probability
2 * 0.5^8 ~= 0.008), so the advantage is non-zero and the optimiser is
genuinely pushed - toward whichever rollouts happened to be labelled 1.
That is precisely the mechanism under test. A constant reward would
instead produce zero variance and no gradient at all, which would test
nothing.

FRESH DRAW PER ROLLOUT - NOT A FIXED LABEL PER PROBLEM
------------------------------------------------------
This matters and is easy to get wrong. Drawing once per PROBLEM and
reusing it would implement Shao et al.'s "incorrect label" condition,
which is a DIFFERENT experiment and produced a substantially larger
effect in their results (+24.1% vs +21.4%). The random-reward condition
draws independently for every rollout, so the same completion can be
rewarded on one sample and not on the next. That is what is implemented
here.

REPRODUCIBILITY - STATED HONESTLY
---------------------------------
veRL evaluates rewards inside Ray workers, so a single module-level RNG
seeded identically in every worker would hand different workers the same
draw sequence. That is not harmful (the rewards stay uncorrelated with
correctness either way) but it is untidy, so the seed is mixed with the
process id. The consequence is that this run is NOT bit-for-bit
reproducible, and the paper should say so rather than imply a determinism
that does not exist. Nothing about this control depends on exact
reproducibility: the claim being tested is about the DISTRIBUTION of the
reward, not about any particular sequence of draws.

USAGE
-----
Identical wiring to src/training/reward_fn.py, only the path changes:

    custom_reward_function.path=<...>/reward_fn_random.py
    custom_reward_function.name=compute_score

Every other training hyperparameter MUST be left untouched, or the
comparison confounds reward-validity with whatever else changed.
"""

from __future__ import annotations

import os
import random

CORRECT_SCORE = 1.0
INCORRECT_SCORE = 0.0

# gamma from Shao et al.'s random-reward condition.
RANDOM_REWARD_P = 0.5

BASE_SEED = 0
_RNG = random.Random(BASE_SEED ^ (os.getpid() * 2654435761 % (2**32)))

# Counted so the run can assert afterwards that the realised rate matched
# the intended one - a silent bug that made this deterministic (e.g. an
# RNG that never advanced) would otherwise be invisible, and would turn
# the control into a constant-reward run that tests nothing.
_calls = 0
_positives = 0


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict
) -> float:
    """
    Reward drawn independently of the completion and of the ground truth.

    `solution_str` and `ground_truth` are accepted because veRL calls
    every custom reward function with this exact four-argument signature,
    and are then DELIBERATELY IGNORED - that is the entire point of the
    control. They are not inspected, parsed, or compared anywhere in this
    module; src.metrics.answer_extraction is intentionally not imported,
    so there is no code path by which correctness could leak into the
    reward.
    """
    global _calls, _positives
    _calls += 1
    hit = _RNG.random() < RANDOM_REWARD_P
    _positives += hit
    return CORRECT_SCORE if hit else INCORRECT_SCORE


def realised_rate() -> dict:
    """
    Realised positive rate in this worker process, for a post-run sanity
    check. Should sit near RANDOM_REWARD_P; a value at 0.0 or 1.0 means
    the RNG was not advancing and the run must be discarded.
    """
    return {
        "calls": _calls,
        "positives": _positives,
        "realised_p": round(_positives / _calls, 4) if _calls else None,
        "intended_p": RANDOM_REWARD_P,
    }
