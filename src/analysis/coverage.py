"""
Per-problem coverage analysis - a CEILING-INDEPENDENT test of the
sharpening claim (PLAN.md Section 4 result box, "Finding 2").

WHY THIS EXISTS
---------------
The Phase 1 result (2026-08-20) reports the sharpening signature via
aggregate pass@64, and that reading is genuinely weak on its own terms -
PLAN.md says so explicitly: base pass@64 is already 0.976-0.990, so
"there is almost no room to detect coverage expansion, so 'no expansion'
and 'cannot tell' are hard to separate here." Both pass@64 deltas are in
fact NOT significant (Delta_T/k=64 CI [-0.000, +0.0575]; Delta_E/k=64 CI
[-0.030, +0.0002]). A reviewer reading "pass@64 stays flat" at 0.98
correctly answers: of course it does, there is nowhere to go.

That weakness is a property of the AGGREGATE statistic, not of the data.
The raw records hold per-problem, per-sample outcomes at n=128, which is
strictly more information than pass@64 summarises. This module asks the
sharpening question directly of that finer data:

    Sharpening says RL REWEIGHTS probability mass that the base model
    already had. It therefore predicts that the SET of problems the base
    model can reach at all is unchanged, while the DENSITY of correct
    samples within that set rises.

    Expansion says RL makes previously-unreachable problems reachable.
    It predicts problems move from "never solved in n draws" to
    "sometimes solved".

Those are different, testable predictions about the same records, and
neither depends on where the aggregate ceiling sits. A problem that base
solves 3/128 times and RL solves 40/128 times is invisible to pass@64
(both round to ~1.0) but is exactly the sharpening signature.

WHAT THIS IS NOT
----------------
This is not a replacement for the harder-dataset evaluation. At base
pass@64 ~= 0.98, only a handful of the 50 problems are unsolved, so the
expansion/contraction counts rest on very few problems and their
statistical power is genuinely low - the McNemar test below will usually
be non-significant simply for lack of discordant pairs, and that must be
reported as "underpowered", NOT as "no expansion detected". The
distribution-shift statistics do not suffer this problem and are the
substantive output here.

RELATION TO THE UNBIASED ESTIMATOR
----------------------------------
"Reachable at all" is operationalised as c >= threshold out of n, which
at threshold=1 is the empirical pass@n - a stricter, higher-resolution
question than pass@64 on the same records. It is deliberately NOT the
unbiased pass@k estimator (src/metrics/pass_at_k.py): that estimator
answers "what fraction of k-sized draws would succeed", whereas the
coverage question is about the support of the model's output
distribution, for which the raw count is the direct observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass
class ProblemPair:
    """One problem's (n, c) under both models, paired by problem_idx."""

    problem_idx: int
    n_base: int
    c_base: int
    n_rl: int
    c_rl: int


def pair_by_problem(df, condition: str) -> list[ProblemPair]:
    """
    Build paired per-problem (n, c) records for one condition from the
    two saved records DataFrames.

    `df` must be a DataFrame with columns problem_idx, condition, correct,
    and a `model_kind` column ("base"/"rl") - i.e. the concatenation of
    records_base.parquet and records_rl.parquet with that column added.
    Pairing is by problem_idx, NOT by row order, so a missing or
    reordered problem in one file cannot silently misalign the two models
    (the failure mode bootstrap_delta_ci's docstring warns is the
    caller's responsibility).
    """
    sub = df[df["condition"] == condition]
    out: list[ProblemPair] = []
    grouped = {
        kind: sub[sub["model_kind"] == kind].groupby("problem_idx")["correct"].agg(["count", "sum"])
        for kind in ("base", "rl")
    }
    shared = sorted(set(grouped["base"].index) & set(grouped["rl"].index))
    dropped = (set(grouped["base"].index) | set(grouped["rl"].index)) - set(shared)
    if dropped:
        raise ValueError(
            f"Condition {condition!r}: problem_idx {sorted(dropped)} present for one model but "
            f"not the other. Refusing to compute a paired statistic on unpaired data."
        )
    for idx in shared:
        b, r = grouped["base"].loc[idx], grouped["rl"].loc[idx]
        out.append(
            ProblemPair(
                problem_idx=int(idx),
                n_base=int(b["count"]),
                c_base=int(b["sum"]),
                n_rl=int(r["count"]),
                c_rl=int(r["sum"]),
            )
        )
    return out


def _exact_mcnemar_p(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value on the discordant counts (b, c).

    Implemented directly from the binomial definition rather than pulled
    from scipy, so this has no dependency beyond the standard library:
    under H0 the discordant pairs split Binomial(b+c, 0.5), and the exact
    two-sided p-value is the total probability of splits at least as
    extreme as the observed one. The exact test (not the chi-square
    approximation) is required here because the discordant counts are
    expected to be tiny - single digits - which is precisely where the
    chi-square approximation is invalid.

    Returns 1.0 when there are no discordant pairs at all (nothing to
    test - the models agree on every problem's reachability).
    """
    total = b + c
    if total == 0:
        return 1.0
    observed = min(b, c)
    # Two-sided: sum both tails at or beyond the observed extremity.
    tail = sum(comb(total, i) for i in range(observed + 1))
    p = 2.0 * tail / (2.0**total)
    return min(p, 1.0)


def coverage_analysis(pairs: list[ProblemPair], threshold: int = 1) -> dict:
    """
    Core coverage contrast at one reachability threshold.

    threshold: minimum correct samples for a problem to count as
    "reachable". threshold=1 is the loosest (a single lucky sample counts)
    and is therefore the MOST susceptible to sampling noise; sweeping it
    upward (see coverage_threshold_sweep) shows whether an apparent
    expansion survives a stricter definition or was one fluke draw.

    Returns expansion/contraction counts, the exact McNemar p-value on
    them, and an explicit `underpowered` flag - at a saturated ceiling the
    discordant counts can be so small that non-significance carries no
    information, and that must not be read as evidence of no effect.
    """
    expansion = [p for p in pairs if p.c_base < threshold <= p.c_rl]
    contraction = [p for p in pairs if p.c_rl < threshold <= p.c_base]
    both_reachable = [p for p in pairs if p.c_base >= threshold and p.c_rl >= threshold]
    neither = [p for p in pairs if p.c_base < threshold and p.c_rl < threshold]

    n_exp, n_con = len(expansion), len(contraction)
    p_value = _exact_mcnemar_p(n_exp, n_con)

    return {
        "threshold": threshold,
        "n_problems": len(pairs),
        "expansion_count": n_exp,
        "contraction_count": n_con,
        "net_coverage_change": n_exp - n_con,
        "both_reachable": len(both_reachable),
        "neither_reachable": len(neither),
        "expansion_problem_idx": [p.problem_idx for p in expansion],
        "contraction_problem_idx": [p.problem_idx for p in contraction],
        "mcnemar_p_exact": round(p_value, 4),
        "discordant_total": n_exp + n_con,
        # A conventional rule of thumb for the exact McNemar test: with
        # fewer than ~6 discordant pairs, no split can reach p<0.05
        # two-sided (2*(1/2^5) = 0.0625), so the test CANNOT reject
        # regardless of the data. Flag that explicitly rather than let a
        # null be misread as a finding.
        "underpowered": (n_exp + n_con) < 6,
    }


def coverage_threshold_sweep(pairs: list[ProblemPair], thresholds=(1, 2, 5, 10)) -> list[dict]:
    """
    Coverage contrast at several reachability thresholds.

    An "expansion" that exists only at threshold=1 and vanishes at
    threshold=2 is one lucky sample out of 128, not a new capability.
    Reporting the sweep rather than a single threshold is what makes the
    claim robust to that.
    """
    return [coverage_analysis(pairs, threshold=t) for t in thresholds]


def within_coverage_density_shift(pairs: list[ProblemPair], threshold: int = 1) -> dict:
    """
    The positive half of the sharpening prediction: among problems BOTH
    models can reach, does RL concentrate more mass on correct answers?

    This is the statistic that is genuinely unaffected by the pass@64
    ceiling. Two problems both at pass@64 = 1.0 are indistinguishable to
    the aggregate metric, but if base solves one 3/128 times and RL solves
    it 40/128 times, that is a large, real reweighting - exactly what
    sharpening asserts and what this measures.

    Returns the mean per-problem success RATE for each model over the
    shared reachable set, their difference, and the count of problems
    whose rate rose / fell / tied.
    """
    shared = [p for p in pairs if p.c_base >= threshold and p.c_rl >= threshold]
    if not shared:
        return {"threshold": threshold, "n_shared_reachable": 0, "note": "no shared reachable problems"}

    base_rates = [p.c_base / p.n_base for p in shared]
    rl_rates = [p.c_rl / p.n_rl for p in shared]
    deltas = [r - b for r, b in zip(rl_rates, base_rates)]

    return {
        "threshold": threshold,
        "n_shared_reachable": len(shared),
        "mean_rate_base": round(sum(base_rates) / len(base_rates), 4),
        "mean_rate_rl": round(sum(rl_rates) / len(rl_rates), 4),
        "mean_density_shift": round(sum(deltas) / len(deltas), 4),
        "n_rate_increased": sum(1 for d in deltas if d > 0),
        "n_rate_decreased": sum(1 for d in deltas if d < 0),
        "n_rate_unchanged": sum(1 for d in deltas if d == 0),
        # Sign test on the direction of movement across problems - a
        # distribution-free companion to the mean, robust to a few
        # problems moving a long way.
        "sign_test_p_exact": round(
            _exact_mcnemar_p(sum(1 for d in deltas if d > 0), sum(1 for d in deltas if d < 0)), 4
        ),
    }


def success_count_histogram(pairs: list[ProblemPair], bins: int = 8) -> dict:
    """
    Per-problem success-count distribution for each model, as counts in
    equal-width bins over [0, n]. This is the figure to put in the paper
    in place of a saturated pass@k curve: sharpening predicts the RL
    histogram shifts rightward WITHOUT the zero bin emptying, whereas
    expansion predicts the zero bin empties.

    Assumes a common n across problems (true for this project's
    fixed-n=128 design); raises if that assumption is violated rather
    than silently binning incomparable rates.
    """
    ns = {p.n_base for p in pairs} | {p.n_rl for p in pairs}
    if len(ns) != 1:
        raise ValueError(f"Mixed sample counts {sorted(ns)} - histogram bins would be incomparable.")
    n = ns.pop()
    width = n / bins

    def hist(counts: list[int]) -> list[int]:
        out = [0] * bins
        for c in counts:
            # Final bin is closed on the right so c == n lands in it.
            out[min(int(c / width), bins - 1)] += 1
        return out

    return {
        "n_samples_per_problem": n,
        "bins": bins,
        "bin_edges": [round(i * width, 1) for i in range(bins + 1)],
        "base": hist([p.c_base for p in pairs]),
        "rl": hist([p.c_rl for p in pairs]),
    }


def full_coverage_report(df, conditions=("T", "D", "E")) -> dict:
    """Everything above, per condition, in one JSON-serialisable dict."""
    report = {}
    for condition in conditions:
        pairs = pair_by_problem(df, condition)
        report[condition] = {
            "threshold_sweep": coverage_threshold_sweep(pairs),
            "density_shift": within_coverage_density_shift(pairs, threshold=1),
            "histogram": success_count_histogram(pairs),
        }
    return report
