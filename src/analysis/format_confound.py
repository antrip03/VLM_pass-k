"""
Residual extraction-asymmetry analysis: how much of Delta could still be
format compliance rather than reasoning?

WHY THIS EXISTS
---------------
Commit 43279ab found and fixed a real, result-changing measurement bug:
the extractor rejected decorated markers ("#### <18>", "#### \\$18"), the
rejection rate was ASYMMETRIC between models (base 21.2% vs RL 5.9% on
T), and because src/training/reward_fn.py shares that extractor, RL had
been explicitly rewarded for emitting cleanly-parseable answers.
Uncorrected, Delta_text read +0.099 instead of +0.059 - roughly 40% of the
apparent reasoning gain was format compliance.

The fix tolerated the known decorations, but PLAN.md records that the
problem is NOT fully closed:

    "A residual asymmetry remains (E still parses worse than T: base
     21.0% vs 13.4%), so part of the T->E level gap is still measurement
     rather than capability."

So a reviewer's obvious question - "how do you know the REMAINING +5.9%
isn't also format compliance?" - is currently unanswered. Reporting the
residual rate does not answer it. This module answers it two ways:

  1. SENSITIVITY (hypothetical): what fraction of still-unparsed
     completions would have to be secretly correct for Delta to reach
     zero? If that fraction exceeds 1.0, no amount of hidden correctness
     can explain the result away and the finding is robust by
     construction.

  2. EMPIRICAL (measured): the completions are saved, so we do not have
     to hypothesise. Search the unparsed completions for the ground-truth
     value and measure directly how often it is present. That is an
     UPPER bound on the hidden-correctness rate, not an estimate of it -
     a number can appear incidentally in working without being the
     model's final answer - and it is reported as a bound throughout.

Together these convert "a residual asymmetry remains" from an
acknowledged weakness into a quantified, bounded one.
"""

from __future__ import annotations

import re

# Matches the ground-truth value as a standalone number, tolerating
# thousands separators and a trailing ".0" - deliberately NOT a bare
# substring search, which would count "18" as present inside "180" and
# badly inflate the upper bound this module reports.
def _ground_truth_present(text: str, ground_truth: str, tail_chars: int = 200) -> bool:
    """
    Is the ground-truth value present as a standalone number near the end
    of the completion?

    Restricted to the tail because a value appearing early is very likely
    an intermediate quantity in the working rather than the final answer;
    restricting to the tail keeps this an upper bound that is at least
    plausibly tight. 200 chars is deliberately looser than the 120 used
    in scripts/diagnose_no_answer.py, since that script was asking a
    different question (is there any digit at all, i.e. is this
    truncation?) and a looser window here makes the bound safely
    conservative.
    """
    if not ground_truth:
        return False
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    try:
        value = float(str(ground_truth).replace(",", ""))
    except (TypeError, ValueError):
        return False

    for token in re.findall(r"-?[\d,]+(?:\.\d+)?", tail):
        try:
            if float(token.replace(",", "")) == value:
                return True
        except ValueError:
            continue
    return False


def _raw_rates(df, condition: str) -> dict:
    """
    UNROUNDED per-model counts and rates for one condition.

    Kept separate from parse_rates() deliberately. parse_rates() rounds
    for human-readable reporting, but the tipping point below divides a
    delta by the between-model unparsed GAP, and that gap is small - on
    the real Phase 1 data it is on the order of 0.05-0.15. Dividing a
    4-decimal-rounded numerator by a 4-decimal-rounded small denominator
    magnifies the rounding error into the answer: this smoke-tested at
    p*=2.078 instead of the true 2.0781, and the discrepancy grows as the
    gap shrinks. Statistics that feed a verdict are computed from raw
    counts; rounding happens only at the point of display.
    """
    out = {}
    for kind in ("base", "rl"):
        sub = df[(df["condition"] == condition) & (df["model_kind"] == kind)]
        total = len(sub)
        if total == 0:
            out[kind] = None
            continue
        unparsed = int(sub["extracted_answer"].isna().sum())
        correct = int(sub["correct"].sum())
        out[kind] = {
            "total_samples": total,
            "unparsed": unparsed,
            "correct": correct,
            "unparsed_rate": unparsed / total,
            "measured_accuracy": correct / total,
        }
    return out


def parse_rates(df, conditions=("T", "D", "E")) -> dict:
    """
    Per (model, condition): total samples, unparsed count, unparsed rate,
    and the between-model asymmetry that drives any residual confound.

    A POSITIVE `asymmetry` means base is unparsed more often than RL,
    which is the direction that INFLATES Delta (base loses credit RL
    keeps). A negative value would deflate it.

    Values here are ROUNDED for reporting - see _raw_rates() for why the
    computations do not use these.
    """
    out = {}
    for condition in conditions:
        raw = _raw_rates(df, condition)
        entry = {}
        for kind in ("base", "rl"):
            r = raw[kind]
            entry[kind] = (
                {
                    "total_samples": r["total_samples"],
                    "unparsed": r["unparsed"],
                    "unparsed_rate": round(r["unparsed_rate"], 4),
                    "measured_accuracy": round(r["measured_accuracy"], 4),
                }
                if r
                else {"total_samples": 0, "unparsed": 0, "unparsed_rate": None, "measured_accuracy": None}
            )
        if raw["base"] and raw["rl"]:
            entry["asymmetry_base_minus_rl"] = round(
                raw["base"]["unparsed_rate"] - raw["rl"]["unparsed_rate"], 4
            )
        out[condition] = entry
    return out


def sensitivity_to_hidden_correctness(df, condition: str) -> dict:
    """
    At what hidden-correctness rate p does the measured Delta vanish?

    Model: suppose a fraction p of each model's UNPARSED completions were
    in fact correct but unreadable by the extractor. Assuming the same p
    for both models - the neutral assumption; assuming a higher p for base
    is what an adversarial reviewer would propose, and is covered by the
    worst-case bound below - the corrected accuracies are

        acc'(model) = (correct + p * unparsed) / total

    so the corrected delta is linear in p:

        Delta'(p) = Delta_measured + p * (u_rl - u_base)

    where u_* are unparsed RATES. Setting Delta'(p) = 0 gives the tipping
    point

        p* = Delta_measured / (u_base - u_rl)

    which only exists in [0, 1] when base is the more-unparsed model.
    p* > 1 means the confound CANNOT explain the result away even if every
    unparsed base completion were secretly correct.

    Also reports the adversarial worst case (every unparsed BASE
    completion correct, every unparsed RL completion wrong), which is the
    hardest bound a reviewer can demand.
    """
    raw = _raw_rates(df, condition)
    if not raw["base"] or not raw["rl"]:
        return {"condition": condition, "note": "missing records for one model - cannot compute"}
    u_base = raw["base"]["unparsed_rate"]
    u_rl = raw["rl"]["unparsed_rate"]
    delta = raw["rl"]["measured_accuracy"] - raw["base"]["measured_accuracy"]
    gap = u_base - u_rl

    if gap <= 0:
        tipping = None
        verdict = (
            "RL is unparsed at least as often as base, so residual extraction failure "
            "cannot be inflating Delta - if anything it deflates it."
        )
    else:
        tipping = delta / gap
        if tipping > 1.0:
            verdict = (
                f"ROBUST: even if 100% of unparsed completions were secretly correct, Delta "
                f"stays positive (tipping point p*={tipping:.2f} exceeds 1.0)."
            )
        else:
            verdict = (
                f"SENSITIVE: if {tipping:.1%} of unparsed completions were secretly correct, "
                f"Delta would vanish. Compare against the measured upper bound."
            )

    worst_case_delta = delta - u_base * 1.0 + u_rl * 0.0
    return {
        "condition": condition,
        "delta_measured": round(delta, 4),
        "unparsed_rate_base": round(u_base, 4),
        "unparsed_rate_rl": round(u_rl, 4),
        "asymmetry": round(gap, 4),
        "tipping_point_p": round(tipping, 4) if tipping is not None else None,
        "delta_worst_case": round(worst_case_delta, 4),
        "verdict": verdict,
    }


def empirical_hidden_correctness_bound(df, condition: str, tail_chars: int = 200) -> dict:
    """
    Measured UPPER BOUND on the hidden-correctness rate, per model.

    We have the completions, so the hypothetical p above does not have to
    stay hypothetical. For every unparsed completion, check whether the
    ground-truth value appears as a standalone number in the tail. The
    resulting rate is an upper bound because a matching number may be an
    intermediate result in the working rather than an intended final
    answer - so this OVERSTATES how much correctness is hidden, which is
    the safe direction for a robustness argument.

    Reported alongside the tipping point: if the measured upper bound is
    comfortably below p*, the residual confound demonstrably cannot
    account for Delta.
    """
    out = {"condition": condition, "tail_chars": tail_chars}
    for kind in ("base", "rl"):
        sub = df[
            (df["condition"] == condition)
            & (df["model_kind"] == kind)
            & (df["extracted_answer"].isna())
        ]
        total_unparsed = len(sub)
        if total_unparsed == 0:
            out[kind] = {"unparsed": 0, "gt_present": 0, "upper_bound_rate": None}
            continue
        hits = sum(
            1
            for _, row in sub.iterrows()
            if _ground_truth_present(str(row["completion"]), str(row["ground_truth"]), tail_chars)
        )
        out[kind] = {
            "unparsed": total_unparsed,
            "gt_present": hits,
            "upper_bound_rate": round(hits / total_unparsed, 4),
        }
    return out


def full_format_confound_report(df, conditions=("T", "D", "E")) -> dict:
    """
    Everything above per condition, plus a combined verdict comparing the
    measured upper bound against the tipping point - the sentence that
    actually goes in the paper.
    """
    report = {"parse_rates": parse_rates(df, conditions), "per_condition": {}}
    for condition in conditions:
        sens = sensitivity_to_hidden_correctness(df, condition)
        emp = empirical_hidden_correctness_bound(df, condition)
        combined = None
        p_star, bound = sens["tipping_point_p"], emp.get("base", {}).get("upper_bound_rate")
        if p_star is not None and bound is not None:
            if bound < p_star:
                combined = (
                    f"Delta_{condition} survives: the measured upper bound on hidden correctness "
                    f"({bound:.1%}) is below the tipping point ({p_star:.1%}) needed to nullify it."
                )
            else:
                combined = (
                    f"Delta_{condition} is NOT established as robust: the measured upper bound "
                    f"({bound:.1%}) reaches the tipping point ({p_star:.1%}). Manual review of "
                    f"unparsed completions is required before reporting this Delta."
                )
        report["per_condition"][condition] = {
            "sensitivity": sens,
            "empirical_upper_bound": emp,
            "verdict": combined,
        }
    return report
