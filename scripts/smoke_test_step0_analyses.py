"""
Synthetic-data smoke test for the Step 0 analyses
(src/analysis/coverage.py, src/analysis/format_confound.py,
src/controls/spurious_correctness.py).

WHY SYNTHETIC AND NOT THE REAL RECORDS
--------------------------------------
The real Phase 1 records give no way to check that these analyses are
CORRECT - we do not independently know the true expansion count or the
true tipping point in that data, so running against it can only confirm
the code does not crash. Here the data is constructed with hand-computed
answers, so every assertion below is a real correctness check on the
maths, not a smoke check on the plumbing.

This matters more than usual for this project: commit 43279ab was a
measurement bug that produced a plausible-looking but 40%-inflated
headline number, invisibly. Analyses that defend against that class of
error must themselves be verified against known answers.

Runs locally in about a second. No GPU, no Modal, no credentials.

    python3 scripts/smoke_test_step0_analyses.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analysis.coverage import (
    coverage_analysis,
    pair_by_problem,
    success_count_histogram,
    within_coverage_density_shift,
)
from src.analysis.format_confound import (
    empirical_hidden_correctness_bound,
    parse_rates,
    sensitivity_to_hidden_correctness,
)
from src.controls.difficulty_matched import per_problem_deltas, stratified_comparison
from src.controls.spurious_correctness import (
    sample_correct_traces,
    score_blinded_review,
    write_blinded_review_pack,
)

N_SAMPLES = 32
N_PROBLEMS = 20

# --- The designed ground truth ------------------------------------------
# base: problems 0-15 -> 16 correct; 16 -> 4; 17 -> 2; 18,19 -> 0
# rl:   problems 0-15 -> 24 correct; 16 -> 8; 17 -> 0; 18 -> 3; 19 -> 0
# So at threshold=1 there is exactly ONE expansion (problem 18: 0 -> 3)
# and exactly ONE contraction (problem 17: 2 -> 0).
BASE_CORRECT = {**{i: 16 for i in range(16)}, 16: 4, 17: 2, 18: 0, 19: 0}
RL_CORRECT = {**{i: 24 for i in range(16)}, 16: 8, 17: 0, 18: 3, 19: 0}

# Unparsed counts, placed only among INCORRECT samples (an unparsed
# completion cannot have been scored correct). 20% for base, 10% for RL -
# the asymmetric direction that inflates Delta.
BASE_UNPARSED_TOTAL = 128  # 128/640 = 0.20
RL_UNPARSED_TOTAL = 64  # 64/640 = 0.10


def build_records(condition: str) -> list[dict]:
    rows = []
    for kind, correct_map, unparsed_total in (
        ("base", BASE_CORRECT, BASE_UNPARSED_TOTAL),
        ("rl", RL_CORRECT, RL_UNPARSED_TOTAL),
    ):
        unparsed_budget = unparsed_total
        for pidx in range(N_PROBLEMS):
            n_correct = correct_map[pidx]
            for sidx in range(N_SAMPLES):
                is_correct = sidx < n_correct
                gt = str(100 + pidx)
                if is_correct:
                    completion, extracted = f"reasoning here\n#### {gt}", gt
                elif unparsed_budget > 0:
                    unparsed_budget -= 1
                    # Half of the unparsed ones contain the ground-truth
                    # value in the tail, so the measured upper bound on
                    # hidden correctness should come out at exactly 0.5.
                    if unparsed_budget % 2 == 0:
                        completion = f"working... the total comes to {gt} dollars altogether"
                    else:
                        completion = "working... I am not sure how to finish this one"
                    extracted = None
                else:
                    wrong = str(900 + pidx)
                    completion, extracted = f"reasoning here\n#### {wrong}", wrong
                rows.append(
                    {
                        "problem_idx": pidx,
                        "question": f"Question text for problem {pidx}",
                        "ground_truth": gt,
                        "condition": condition,
                        "sample_idx": sidx,
                        "completion": completion,
                        "extracted_answer": extracted,
                        "transcription": (
                            f"Question text for problem {pidx}" if condition == "D" else None
                        ),
                        "correct": is_correct,
                        "model_kind": kind,
                    }
                )
    return rows


def approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) < tol


DIFF_N = 40
DIFF_PROBLEMS = 40


def build_difficulty_frame(extra_e_gain: float) -> "pd.DataFrame":
    """
    A frame designed to have the SAME aggregate signature as the real
    Phase 1 result - condition E sits at a lower base rate and shows a
    larger aggregate Delta - under two opposite underlying truths:

      extra_e_gain = 0.0   PURE FLOOR EFFECT. Delta is a function of the
                           observation's OWN base rate and nothing else,
                           identical for T and E. E only looks better
                           because it is harder.

      extra_e_gain = 0.10  REAL MODALITY EFFECT. At equal base rate, E
                           still gains 10 points more than T.

Both produce Delta_E > Delta_T in aggregate. Only a control that
    actually holds difficulty constant can separate them - which is
    precisely what this test verifies.
    """
    rows = []
    for pidx in range(DIFF_PROBLEMS):
        # T base rates spread 0.35..0.95; E is 0.25 lower for the same
        # problem, mirroring the real T=0.609 vs E=0.496 offset.
        base_t = 0.35 + 0.60 * pidx / (DIFF_PROBLEMS - 1)
        base_e = base_t - 0.25
        for condition, base in (("T", base_t), ("E", base_e)):
            delta = 0.30 * (1.0 - base) + (extra_e_gain if condition == "E" else 0.0)
            rl = min(base + delta, 1.0)
            for kind, rate in (("base", base), ("rl", rl)):
                n_correct = round(rate * DIFF_N)
                for sidx in range(DIFF_N):
                    rows.append(
                        {
                            "problem_idx": pidx,
                            "question": f"q{pidx}",
                            "ground_truth": "1",
                            "condition": condition,
                            "sample_idx": sidx,
                            "completion": "#### 1" if sidx < n_correct else "#### 2",
                            "extracted_answer": "1" if sidx < n_correct else "2",
                            "transcription": None,
                            "correct": sidx < n_correct,
                            "model_kind": kind,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    rows = []
    for cond in ("T", "D", "E"):
        rows.extend(build_records(cond))
    df = pd.DataFrame(rows)
    print(f"Built synthetic frame: {len(df):,} rows")

    failures: list[str] = []

    def check(label: str, got, want) -> None:
        ok = approx(got, want) if isinstance(want, float) else got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: got={got} want={want}")
        if not ok:
            failures.append(label)

    # ---------------- coverage ----------------
    print("\n=== coverage_analysis (threshold=1) ===")
    pairs = pair_by_problem(df, "T")
    check("n_problems", len(pairs), N_PROBLEMS)
    cov1 = coverage_analysis(pairs, threshold=1)
    check("expansion_count", cov1["expansion_count"], 1)
    check("contraction_count", cov1["contraction_count"], 1)
    check("expansion is problem 18", cov1["expansion_problem_idx"], [18])
    check("contraction is problem 17", cov1["contraction_problem_idx"], [17])
    check("both_reachable", cov1["both_reachable"], 17)
    check("neither_reachable", cov1["neither_reachable"], 1)
    # b=c=1 -> perfectly balanced discordance -> exact two-sided p clamps to 1.0
    check("mcnemar_p_exact", cov1["mcnemar_p_exact"], 1.0)
    check("underpowered (2 discordant < 6)", cov1["underpowered"], True)

    print("\n=== coverage_analysis (threshold=5) ===")
    cov5 = coverage_analysis(pairs, threshold=5)
    # Only problem 16 crosses upward (4 -> 8); problem 18 (0 -> 3) does not
    # reach 5, and problem 17 (2 -> 0) was never above 5 to begin with.
    check("expansion_count", cov5["expansion_count"], 1)
    check("expansion is problem 16", cov5["expansion_problem_idx"], [16])
    check("contraction_count", cov5["contraction_count"], 0)

    print("\n=== within_coverage_density_shift ===")
    dens = within_coverage_density_shift(pairs, threshold=1)
    # shared reachable = problems 0..16 (17 problems)
    # base mean rate = (16*16 + 4)/32/17 = 8.125/17  = 0.477941
    # rl   mean rate = (16*24 + 8)/32/17 = 12.25/17  = 0.720588
    check("n_shared_reachable", dens["n_shared_reachable"], 17)
    check("mean_rate_base", dens["mean_rate_base"], 0.4779)
    check("mean_rate_rl", dens["mean_rate_rl"], 0.7206)
    check("mean_density_shift", dens["mean_density_shift"], 0.2426)
    check("n_rate_increased", dens["n_rate_increased"], 17)
    check("n_rate_decreased", dens["n_rate_decreased"], 0)

    print("\n=== success_count_histogram ===")
    hist = success_count_histogram(pairs, bins=8)
    check("n_samples_per_problem", hist["n_samples_per_problem"], N_SAMPLES)
    # base zero-bin holds problems 17(c=2),18(c=0),19(c=0) -> c in [0,4)
    check("base zero-bin count", hist["base"][0], 3)
    # rl zero-bin holds 17(c=0), 18(c=3), 19(c=0) -> also 3
    check("rl zero-bin count", hist["rl"][0], 3)
    check("histogram totals match problem count", sum(hist["base"]), N_PROBLEMS)

    # ---------------- format confound ----------------
    print("\n=== parse_rates ===")
    rates = parse_rates(df, conditions=("T",))["T"]
    check("base unparsed_rate", rates["base"]["unparsed_rate"], 0.2)
    check("rl unparsed_rate", rates["rl"]["unparsed_rate"], 0.1)
    check("asymmetry", rates["asymmetry_base_minus_rl"], 0.1)
    # base correct = 16*16 + 4 + 2 = 262 of 640
    check("base measured_accuracy", rates["base"]["measured_accuracy"], 0.4094)
    # rl correct = 16*24 + 8 + 3 = 395 of 640
    check("rl measured_accuracy", rates["rl"]["measured_accuracy"], 0.6172)

    print("\n=== sensitivity_to_hidden_correctness ===")
    sens = sensitivity_to_hidden_correctness(df, "T")
    # delta = 0.6171875 - 0.409375 = 0.2078125 ; gap = 0.1
    # p* = 0.2078125 / 0.1 = 2.078125  -> > 1 -> ROBUST
    check("delta_measured", sens["delta_measured"], 0.2078)
    check("tipping_point_p", sens["tipping_point_p"], 2.0781)
    check("verdict says ROBUST", sens["verdict"].startswith("ROBUST"), True)
    # worst case: delta - u_base = 0.2078125 - 0.2 = 0.0078125
    check("delta_worst_case", sens["delta_worst_case"], 0.0078)

    print("\n=== empirical_hidden_correctness_bound ===")
    emp = empirical_hidden_correctness_bound(df, "T")
    # By construction exactly half the unparsed completions carry the
    # ground-truth value in their tail.
    check("base unparsed count", emp["base"]["unparsed"], 128)
    check("base upper_bound_rate", emp["base"]["upper_bound_rate"], 0.5)
    check("rl upper_bound_rate", emp["rl"]["upper_bound_rate"], 0.5)
    # 0.5 measured bound < 2.078 tipping point -> confound cannot explain Delta
    check("bound below tipping point", emp["base"]["upper_bound_rate"] < sens["tipping_point_p"], True)

    # ---------------- spurious-correctness pack ----------------
    print("\n=== spurious_correctness blinding ===")
    traces = sample_correct_traces(df, "T", n_per_model=10)
    check("traces drawn", len(traces), 20)
    check("balanced across models", sum(1 for t in traces if t["model_kind"] == "rl"), 10)
    check(
        "spread across distinct problems",
        len({t["problem_idx"] for t in traces if t["model_kind"] == "base"}) >= 10,
        True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        pack = write_blinded_review_pack(traces, tmp)
        check("pack size", pack["n_traces"], 20)
        review_rows = [
            json.loads(line)
            for line in Path(pack["review_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        check("no model_kind leaked into review file",
              all("model_kind" not in r for r in review_rows), True)
        check("all verdicts start null", all(r["verdict"] is None for r in review_rows), True)

        # Refuses to score an incomplete review rather than silently
        # treating unreviewed rows as sound.
        try:
            score_blinded_review(pack["review_path"], pack["key_path"])
            check("refuses incomplete review", "no error raised", "ValueError")
        except ValueError:
            check("refuses incomplete review", True, True)

        # Fill verdicts and score: mark every 5th trace spurious.
        for i, r in enumerate(review_rows):
            r["verdict"] = "spurious" if i % 5 == 0 else "sound"
        Path(pack["review_path"]).write_text(
            "\n".join(json.dumps(r) for r in review_rows), encoding="utf-8"
        )
        scored = score_blinded_review(pack["review_path"], pack["key_path"])
        total_spurious = sum(scored[k]["spurious"] for k in ("base", "rl") if k in scored)
        check("scored spurious total", total_spurious, 4)

    # ---------------- difficulty-matched / floor effect ----------------
    # Two designed worlds with the SAME aggregate signature (Delta_E >
    # Delta_T, because E sits at a lower base rate) but opposite truths.
    # A correct control must tell them apart.
    print("\n=== difficulty_matched: FLOOR-EFFECT world ===")
    floor_df = build_difficulty_frame(extra_e_gain=0.0)
    floor = stratified_comparison(floor_df, ("T", "E"), n_bins=4)
    print(f"  weighted gap = {floor['weighted_mean_gap']}  ({floor['verdict'][:60]}...)")
    agg_t = sum(p.delta for p in per_problem_deltas(floor_df, "T"))
    agg_e = sum(p.delta for p in per_problem_deltas(floor_df, "E"))
    check("aggregate Delta_E > Delta_T (the misleading signature)", agg_e > agg_t, True)
    check("within-bin gap collapses to ~0", abs(floor["weighted_mean_gap"]) <= 0.02, True)
    check("verdict identifies FLOOR EFFECT", floor["verdict"].startswith("FLOOR EFFECT"), True)

    print("\n=== difficulty_matched: REAL-EFFECT world ===")
    real_df = build_difficulty_frame(extra_e_gain=0.10)
    real = stratified_comparison(real_df, ("T", "E"), n_bins=4)
    print(f"  weighted gap = {real['weighted_mean_gap']}  ({real['verdict'][:60]}...)")
    check("within-bin gap survives at ~+0.10", real["weighted_mean_gap"] > 0.05, True)
    check("verdict identifies ORDERING SURVIVES",
          real["verdict"].startswith("ORDERING SURVIVES"), True)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        raise SystemExit(1)
    print("ALL STEP 0 ANALYSIS CHECKS PASSED")


if __name__ == "__main__":
    main()
