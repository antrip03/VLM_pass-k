"""
Difficulty-matched text control (PLAN.md Section 4 item 9, Section 10
item 2): is Delta_pixel > Delta_text a real modality effect, or a floor
effect?

THE OBJECTION, WHICH THE PROJECT ALREADY CONCEDES
-------------------------------------------------
PLAN.md's Phase 1 result box states it as a known limit, unprompted:

    "Delta_pixel > Delta_text is partly a floor effect. E starts far
     lower (0.496 vs 0.609), so it has more headroom. Do NOT claim
     images benefit *more* from text-only RL."

Right now that is a caveat the paper can only STATE. This module lets it
be TESTED, which is the difference between an acknowledged weakness and a
controlled result.

TWO TESTS, AND WHY BOTH
-----------------------
1. MATCHED SUBSET - the literal design PLAN.md specifies. Select the
   text problems whose base pass@1 sits near condition E's overall base
   rate, and recompute Delta_text on just those. Directly interpretable,
   but throws away most of the 50 problems and is therefore badly
   underpowered at this sample size.

2. STRATIFIED COMPARISON - the same idea using ALL the data. Bin problems
   by base difficulty and compare Delta_T against Delta_E WITHIN each
   bin. Because T, D and E are evaluated on the SAME problems, each bin
   holds difficulty roughly constant while modality varies, which is
   exactly the contrast the control wants - and it keeps every problem
   rather than discarding those outside a narrow band.

   If Delta_E exceeds Delta_T within bins, the ordering survives the
   floor-effect explanation. If the two converge once difficulty is held
   constant, the ordering WAS a floor effect, and the paper should say so
   - which is the honest outcome PLAN.md already leans toward.

Both are reported, with the underpowered one flagged as such, rather than
quietly reporting whichever looks better.

DIFFICULTY IS MEASURED ON THE BASE MODEL ONLY
---------------------------------------------
Binning by a quantity computed from the RL model would condition on the
outcome and bias the contrast. Base per-problem pass@1 is a property of
the problem and the untrained model, fixed before any of this
intervention - so it is a legitimate stratifier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemDelta:
    """One problem's base rate, RL rate and their difference, per condition."""

    problem_idx: int
    condition: str
    base_rate: float
    rl_rate: float

    @property
    def delta(self) -> float:
        return self.rl_rate - self.base_rate


def per_problem_deltas(df, condition: str) -> list[ProblemDelta]:
    """
    Per-problem base/RL success rates for one condition.

    Uses the raw success RATE (c/n), which is the unbiased pass@1
    estimator, so these numbers are directly comparable with the headline
    pass@1 figures rather than being a different statistic that happens to
    look similar.
    """
    sub = df[df["condition"] == condition]
    rates = {}
    for kind in ("base", "rl"):
        g = sub[sub["model_kind"] == kind].groupby("problem_idx")["correct"]
        rates[kind] = {int(i): float(s) / int(c) for i, c, s in zip(g.count().index, g.count(), g.sum())}

    shared = sorted(set(rates["base"]) & set(rates["rl"]))
    return [
        ProblemDelta(idx, condition, rates["base"][idx], rates["rl"][idx]) for idx in shared
    ]


def matched_subset(
    df,
    target_condition: str = "E",
    matched_condition: str = "T",
    tolerance: float = 0.10,
) -> dict:
    """
    PLAN.md's literal design: recompute Delta on the text problems whose
    base difficulty matches condition E's overall base rate.

    `tolerance` is the half-width of the accepted band around E's base
    pass@1. Widening it buys statistical power at the cost of match
    quality; both the band and the resulting subset size are reported so
    the trade-off is visible rather than buried in a default.
    """
    target = per_problem_deltas(df, target_condition)
    matched = per_problem_deltas(df, matched_condition)
    if not target or not matched:
        return {"note": "missing records for one of the conditions"}

    target_base_mean = sum(p.base_rate for p in target) / len(target)
    lo, hi = target_base_mean - tolerance, target_base_mean + tolerance
    subset = [p for p in matched if lo <= p.base_rate <= hi]

    if not subset:
        return {
            "target_condition": target_condition,
            "matched_condition": matched_condition,
            "target_base_mean": round(target_base_mean, 4),
            "band": [round(lo, 4), round(hi, 4)],
            "n_matched": 0,
            "note": "no problems fell in the band - widen tolerance",
        }

    subset_delta = sum(p.delta for p in subset) / len(subset)
    target_delta = sum(p.delta for p in target) / len(target)
    full_delta = sum(p.delta for p in matched) / len(matched)

    return {
        "target_condition": target_condition,
        "matched_condition": matched_condition,
        "target_base_mean": round(target_base_mean, 4),
        "band": [round(lo, 4), round(hi, 4)],
        "n_matched": len(subset),
        "n_total": len(matched),
        "matched_base_mean": round(sum(p.base_rate for p in subset) / len(subset), 4),
        f"delta_{matched_condition}_full": round(full_delta, 4),
        f"delta_{matched_condition}_matched_subset": round(subset_delta, 4),
        f"delta_{target_condition}": round(target_delta, 4),
        "gap_after_matching": round(target_delta - subset_delta, 4),
        # With 50 problems and a +-0.10 band, the matched subset is
        # routinely under 20 problems. Say so, rather than let a noisy
        # point estimate be read as a result.
        "underpowered": len(subset) < 20,
        "interpretation": (
            f"If gap_after_matching is near zero, the Delta_{target_condition} > "
            f"Delta_{matched_condition} ordering is explained by baseline difficulty (a floor "
            f"effect) rather than by modality. If it remains clearly positive, the ordering "
            f"survives the floor-effect explanation."
        ),
    }


def stratified_comparison(df, conditions=("T", "E"), n_bins: int = 4) -> dict:
    """
    The better-powered version: compare Delta across conditions WITHIN
    bins of base difficulty, using every problem.

    BINNING IS ON EACH OBSERVATION'S OWN BASE RATE - a correction to the
    obvious-but-wrong approach. Binning by a reference condition's
    difficulty (e.g. bin everything by its Condition-T base rate) does
    NOT hold difficulty constant, because the same problem is
    systematically harder in E than in T: a problem in the "mid
    difficulty" T bin sits at a lower base rate in E, so the within-bin
    contrast would still be comparing unequal difficulties and would
    report a spurious gap of roughly the size of that offset.

    Instead every (problem, condition) observation is pooled and binned by
    ITS OWN base rate, so a bin contains observations of genuinely
    comparable difficulty regardless of which condition they came from.

    The cost of that correction, stated plainly: within a bin the T and E
    observations are no longer the same problems, so the contrast is
    unpaired. That is the right trade here - the question being asked is
    about difficulty, not about problem identity, and the paired version
    cannot answer it.
    """
    per_cond = {c: per_problem_deltas(df, c) for c in conditions}
    if any(not v for v in per_cond.values()):
        return {"note": "missing records for one or more conditions"}

    # Pool every observation and bin on its own base rate.
    pooled = [p for c in conditions for p in per_cond[c]]
    ordered = sorted(p.base_rate for p in pooled)
    edges = [
        ordered[min(int(i * len(ordered) / n_bins), len(ordered) - 1)] for i in range(1, n_bins)
    ]

    def bin_of(rate: float) -> int:
        for i, e in enumerate(edges):
            if rate < e:
                return i
        return n_bins - 1

    bins = []
    for b in range(n_bins):
        members = [p for p in pooled if bin_of(p.base_rate) == b]
        entry = {
            "bin": b,
            "n_observations": len(members),
            "base_rate_range": (
                [round(min(p.base_rate for p in members), 4),
                 round(max(p.base_rate for p in members), 4)]
                if members
                else None
            ),
        }
        for c in conditions:
            mine = [p for p in members if p.condition == c]
            entry[c] = {
                "n": len(mine),
                "mean_delta": round(sum(p.delta for p in mine) / len(mine), 4) if mine else None,
                "mean_base_rate": (
                    round(sum(p.base_rate for p in mine) / len(mine), 4) if mine else None
                ),
            }
        if len(conditions) == 2:
            a, bb = conditions
            if entry[a]["mean_delta"] is not None and entry[bb]["mean_delta"] is not None:
                entry["gap"] = round(entry[bb]["mean_delta"] - entry[a]["mean_delta"], 4)
        bins.append(entry)

    # Weight by the smaller per-condition count in each bin: a bin with 30
    # T observations and 1 E observation should not dominate the average
    # on the strength of an n=1 mean.
    valid = [
        b for b in bins
        if b.get("gap") is not None and min(b[conditions[0]]["n"], b[conditions[1]]["n"]) > 0
    ]
    weights = [min(b[conditions[0]]["n"], b[conditions[1]]["n"]) for b in valid]
    weighted_gap = (
        sum(b["gap"] * w for b, w in zip(valid, weights)) / sum(weights) if sum(weights) else None
    )
    positive_bins = sum(1 for b in valid if b["gap"] > 0)

    verdict = None
    if weighted_gap is not None:
        if positive_bins == len(valid) and weighted_gap > 0.02:
            verdict = (
                f"ORDERING SURVIVES: Delta_{conditions[1]} exceeds Delta_{conditions[0]} in all "
                f"{len(valid)} difficulty bins (weighted gap {weighted_gap:+.4f}). The ordering "
                f"is not explained by baseline difficulty alone."
            )
        elif abs(weighted_gap) <= 0.02:
            verdict = (
                f"FLOOR EFFECT: once difficulty is held constant the gap collapses to "
                f"{weighted_gap:+.4f}. The ordering should be reported as a baseline-level "
                f"artifact, exactly as PLAN.md's result box anticipated."
            )
        else:
            verdict = (
                f"MIXED: weighted gap {weighted_gap:+.4f}, positive in {positive_bins}/{len(valid)} "
                f"bins. Report per-bin rather than as a single ordering claim."
            )

    return {
        "conditions": list(conditions),
        "n_bins": n_bins,
        "binned_on": "each observation's own base rate (pooled across conditions)",
        "bin_edges": [round(e, 4) for e in edges],
        "bins": bins,
        "weighted_mean_gap": round(weighted_gap, 4) if weighted_gap is not None else None,
        "bins_favouring_second_condition": f"{positive_bins}/{len(valid)}",
        "verdict": verdict,
    }


def full_difficulty_report(df) -> dict:
    """Both tests, for the two orderings the paper actually claims."""
    return {
        "matched_subset_E_vs_T": matched_subset(df, "E", "T"),
        "matched_subset_D_vs_T": matched_subset(df, "D", "T"),
        "stratified_T_vs_E": stratified_comparison(df, ("T", "E")),
        "stratified_T_vs_D": stratified_comparison(df, ("T", "D")),
    }
