"""
Local analysis for the compliance-matching causal control
(scripts/run_compliance_matching_control.py). Free - no GPU.

Compares three things on condition E, all format-agnostic AND strict:

    zero-shot base   (eval_results/records_base.fallback.parquet)
    few-shot  base    (downloaded from the results volume - the new run)
    zero-shot RL      (eval_results/records_rl.fallback.parquet)

THE TEST
--------
1. Did the format exemplar raise compliance, with no training?
   compliance(fewshot base) vs compliance(zeroshot base) vs compliance(RL)

2. Does raising compliance alone collapse the strict Delta?
   strict Delta(fewshot base -> RL) should be SMALLER than
   strict Delta(zeroshot base -> RL), and should move toward the
   format-agnostic Delta - because the compliance gap that inflated it
   has now been closed by prompting, not by training.

3. Is format-agnostic accuracy unchanged by the exemplar?
   fair pass@1(fewshot base) should be close to fair pass@1(zeroshot base).
   If the exemplar also moved REASONING, results are harder to interpret -
   worth knowing, not something to hide.

Usage (after downloading the new run's parquet):
    modal volume get grpo-vlm-phase1b-results \
        compliance_control_records_base_fewshot.parquet \
        eval_results/phase1b/compliance_control_records_base_fewshot.parquet
    python3 scripts/analyze_compliance_control.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.metrics.bootstrap_ci import bootstrap_delta_ci
from src.metrics.pass_at_k import mean_pass_at_k

CONDITION = "E"
FEWSHOT_PATH = "eval_results/phase1b/compliance_control_records_base_fewshot.parquet"


def per_problem(df: pd.DataFrame, col: str):
    g = df.groupby("problem_idx")[col]
    return [(int(c), int(s)) for c, s in zip(g.count(), g.sum())]


def compliance(df: pd.DataFrame) -> float:
    return 1 - df["extracted_answer"].isna().mean()


def main() -> None:
    fewshot = pd.read_parquet(FEWSHOT_PATH)
    zeroshot_base = pd.read_parquet("eval_results/records_base.fallback.parquet")
    zeroshot_base = zeroshot_base[zeroshot_base.condition == CONDITION]
    zeroshot_rl = pd.read_parquet("eval_results/records_rl.fallback.parquet")
    zeroshot_rl = zeroshot_rl[zeroshot_rl.condition == CONDITION]

    # zeroshot records are n=128, fewshot is n=64 - subsample zeroshot to
    # the first 64 samples per problem so the two are directly comparable
    # rather than comparing different sample sizes. This does not
    # discard information asymmetrically: 64 of 128 is a fair, fixed
    # subsample, not a cherry-pick.
    def first_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.groupby("problem_idx", group_keys=False).apply(lambda g: g.head(n))

    zeroshot_base_64 = first_n(zeroshot_base, 64)
    zeroshot_rl_64 = first_n(zeroshot_rl, 64)

    print("=== 1. Format compliance ===")
    comps = {
        "zero-shot base": compliance(zeroshot_base_64),
        "few-shot  base": compliance(fewshot),
        "zero-shot RL  ": compliance(zeroshot_rl_64),
    }
    for label, c in comps.items():
        print(f"  {label}: {c:.4f}")
    moved = comps["few-shot  base"] - comps["zero-shot base"]
    gap_to_rl_before = comps["zero-shot RL  "] - comps["zero-shot base"]
    gap_to_rl_after = comps["zero-shot RL  "] - comps["few-shot  base"]
    print(f"\n  Exemplar raised compliance by {moved:+.4f}")
    print(f"  Gap to RL's compliance: {gap_to_rl_before:.4f} (zero-shot) -> {gap_to_rl_after:.4f} (few-shot)")

    print("\n=== 2. Strict Delta (base -> RL), zero-shot vs few-shot base ===")
    for label, base_df in (("zero-shot base -> RL", zeroshot_base_64), ("few-shot  base -> RL", fewshot)):
        d_strict = bootstrap_delta_ci(per_problem(zeroshot_rl_64, "correct"), per_problem(base_df, "correct"), k=1)
        d_fair = bootstrap_delta_ci(
            per_problem(zeroshot_rl_64, "correct_fallback"), per_problem(base_df, "correct_fallback"), k=1
        )
        print(f"  {label}:")
        print(f"    strict Delta = {d_strict['point_estimate']:+.4f}  "
              f"CI [{d_strict['ci_lower']:+.4f}, {d_strict['ci_upper']:+.4f}]  sig={d_strict['significant']}")
        print(f"    fair   Delta = {d_fair['point_estimate']:+.4f}  "
              f"CI [{d_fair['ci_lower']:+.4f}, {d_fair['ci_upper']:+.4f}]  sig={d_fair['significant']}")

    print("\n=== 3. Did the exemplar change REASONING (fair accuracy), or only formatting? ===")
    fair_zs = mean_pass_at_k(per_problem(zeroshot_base_64, "correct_fallback"), 1)
    fair_fs = mean_pass_at_k(per_problem(fewshot, "correct_fallback"), 1)
    d_reasoning = bootstrap_delta_ci(
        per_problem(fewshot, "correct_fallback"), per_problem(zeroshot_base_64, "correct_fallback"), k=1
    )
    print(f"  zero-shot base fair pass@1: {fair_zs:.4f}")
    print(f"  few-shot  base fair pass@1: {fair_fs:.4f}")
    print(f"  Delta (exemplar effect on reasoning) = {d_reasoning['point_estimate']:+.4f}  "
          f"CI [{d_reasoning['ci_lower']:+.4f}, {d_reasoning['ci_upper']:+.4f}]  sig={d_reasoning['significant']}")
    if d_reasoning["significant"]:
        print("  WARNING: the exemplar also moved reasoning accuracy - interpret Test 2 with this caveat.")
    else:
        print("  Reasoning accuracy unaffected (not significant) - the manipulation is clean.")

    report = {
        "compliance": comps,
        "compliance_gap_to_rl": {"zero_shot": gap_to_rl_before, "few_shot": gap_to_rl_after},
        "exemplar_effect_on_reasoning": {
            "delta": d_reasoning["point_estimate"],
            "ci": [d_reasoning["ci_lower"], d_reasoning["ci_upper"]],
            "significant": d_reasoning["significant"],
        },
    }
    Path("eval_results/compliance_matching_control_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nWrote eval_results/compliance_matching_control_summary.json")


if __name__ == "__main__":
    main()
