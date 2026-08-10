"""
Unbiased pass@k estimator (PLAN.md Section 2.1) - the foundational metric
for the whole experiment. Both the headroom pre-check (Step 4) and the
main evaluation (Steps 6/7) are built on this, which is why it's built
now rather than waiting for Step 7's turn: Step 4 cannot function without
it.

Standard estimator from Chen et al. 2021 ("Evaluating Large Language
Models Trained on Code" / Codex / HumanEval): given n sampled completions
for one problem, of which c are correct, this computes the probability
that at least one of k completions drawn (without replacement) from those
n samples would be correct - NOT simply "was a correct one observed
within the first k samples," which would be a biased, higher-variance
estimate that depends on sample ordering. Requires n >= k.
"""

from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimate for one problem, given n total samples of
    which c were correct.

        pass@k = 1 - C(n-c, k) / C(n, k)

    Computed via the numerically stable product form (avoids overflow
    from computing raw binomial coefficients for large n), following the
    standard HumanEval/Codex implementation:

        1 - prod_{i=n-c+1}^{n} (1 - k / i)

    Raises ValueError if k > n (can't estimate pass@k without at least k
    samples) or if c is out of the valid [0, n] range.
    """
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n}) - need at least k samples")
    if not (0 <= c <= n):
        raise ValueError(f"c ({c}) must be between 0 and n ({n})")

    if n - c < k:
        # Fewer than k incorrect samples exist among the n drawn, so it's
        # certain that any k-sized draw contains at least one correct
        # completion.
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k_for_problems(results_per_problem: list[tuple[int, int]], k: int) -> list[float]:
    """
    Convenience wrapper: given a list of (n, c) pairs, one per problem,
    returns the per-problem pass@k values.
    """
    return [pass_at_k(n, c, k) for n, c in results_per_problem]


def mean_pass_at_k(results_per_problem: list[tuple[int, int]], k: int) -> float:
    """
    Mean pass@k across a set of problems - the headline metric reported
    for a model/condition (e.g. pass@1(base, text)). PLAN.md's
    Delta_text/Delta_pixel are differences of numbers computed this way.
    """
    values = pass_at_k_for_problems(results_per_problem, k)
    if not values:
        raise ValueError("results_per_problem must not be empty")
    return sum(values) / len(values)
