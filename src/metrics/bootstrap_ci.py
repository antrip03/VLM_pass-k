"""
Bootstrap confidence intervals for pass@k metrics (PLAN.md Section 4 item
8: "bootstrap or Bayesian confidence intervals on every reported Delta -
never a bare point estimate"; Section 3 Phase 1 item 8, updated for the
single-seed design: "compute Delta_text and Delta_pixel from the single
seed's results, with a bootstrap/Bayesian CI on each Delta... reflects
sampling noise only, not seed-to-seed variance").

Resamples PROBLEMS (not individual generation samples) with replacement -
the per-problem pass@k value is already a fixed, estimated quantity given
its own n/c; the thing that varies if the eval were re-run is WHICH
problems happened to be in the eval set, so that's the correct
resampling unit for a CI on the mean-across-problems statistic.
"""

from __future__ import annotations

import numpy as np

from src.metrics.pass_at_k import mean_pass_at_k


def bootstrap_pass_at_k_ci(
    results_per_problem: list[tuple[int, int]],
    k: int,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for mean pass@k across problems."""
    rng = np.random.default_rng(seed)
    n_problems = len(results_per_problem)
    if n_problems == 0:
        raise ValueError("results_per_problem must not be empty")

    point_estimate = mean_pass_at_k(results_per_problem, k)

    bootstrap_estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_problems, size=n_problems)
        resampled = [results_per_problem[j] for j in idx]
        bootstrap_estimates[i] = mean_pass_at_k(resampled, k)

    alpha = 1 - confidence
    lower, upper = np.percentile(bootstrap_estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": point_estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n_problems": n_problems,
    }


def bootstrap_delta_ci(
    results_a: list[tuple[int, int]],
    results_b: list[tuple[int, int]],
    k: int,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Paired bootstrap CI for Delta = mean_pass_at_k(A) - mean_pass_at_k(B)
    (e.g. Delta_text = pass@k(pi_RL, T) - pass@k(pi_base, T)). Resamples
    PROBLEM INDICES once per bootstrap iteration and applies the SAME
    resample to both A and B - paired, not independent resampling -
    preserving the real covariance between the two (problems hard for
    one model tend to be hard for the other too). This gives a tighter,
    more honest CI on the difference than resampling A and B
    independently would.

    Requires results_a and results_b to be aligned - the same problem at
    the same index in both lists (e.g. both are the same 25 GSM8K test
    problems, evaluated on pi_base and pi_RL respectively). Only a length
    match is checked here; alignment itself is the caller's
    responsibility.
    """
    if len(results_a) != len(results_b):
        raise ValueError(
            f"results_a and results_b must be the same length (same problems, paired) - "
            f"got {len(results_a)} vs {len(results_b)}"
        )
    n_problems = len(results_a)
    if n_problems == 0:
        raise ValueError("results_a/results_b must not be empty")

    rng = np.random.default_rng(seed)
    point_delta = mean_pass_at_k(results_a, k) - mean_pass_at_k(results_b, k)

    bootstrap_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_problems, size=n_problems)
        resampled_a = [results_a[j] for j in idx]
        resampled_b = [results_b[j] for j in idx]
        bootstrap_deltas[i] = mean_pass_at_k(resampled_a, k) - mean_pass_at_k(resampled_b, k)

    alpha = 1 - confidence
    lower, upper = np.percentile(bootstrap_deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": point_delta,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n_problems": n_problems,
        "significant": bool(not (lower <= 0 <= upper)),  # CI excludes zero
    }
