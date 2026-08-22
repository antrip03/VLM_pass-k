"""
Rescore the committed Phase 1 records with the fallback extractor and
report STRICT vs PERMISSIVE deltas side by side.

This is the script behind the Phase 1b finding that essentially all of
the measured Delta is format compliance rather than reasoning. It costs
nothing: the records store full completion text, so scoring can be redone
offline (the same property commit 43279ab relied on).

    python3 scripts/rescore_with_fallback.py

Outputs `eval_results/rescore_comparison.json` and writes
`records_{base,rl}.fallback.parquet` alongside the originals, carrying two
extra columns:
    correct_fb - correctness under the fallback chain
    method     - which extraction rule fired (audit trail)

WHY BOTH NUMBERS ARE REPORTED, ALWAYS
-------------------------------------
A permissive extractor can always produce a different number, and this
project has already been burned once by an unvalidated extractor. The
strict scoring is what training actually optimised and is not discarded;
the difference between the two IS the quantity of interest, because it
measures how much of the apparent gain is instruction-following.

Do not quote the permissive delta without the validation from
scripts/build_extraction_review_pack.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.metrics.answer_extraction_fallback import is_correct_with_fallback
from src.metrics.bootstrap_ci import bootstrap_delta_ci

CONDITIONS = ("T", "D", "E")


def rescore(kind: str) -> pd.DataFrame:
    path = f"eval_results/records_{kind}.parquet"
    df = pd.read_parquet(path)
    results = [
        is_correct_with_fallback(str(c), str(g))
        for c, g in zip(df["completion"], df["ground_truth"])
    ]
    df["correct_fb"] = [r[0] for r in results]
    df["method"] = [r[1] for r in results]
    out = f"eval_results/records_{kind}.fallback.parquet"
    df.to_parquet(out)
    print(f"  {kind}: rescored {len(df):,} rows -> {out}", flush=True)
    return df


def per_problem(df: pd.DataFrame, condition: str, col: str):
    g = df[df["condition"] == condition].groupby("problem_idx")[col]
    return [(int(c), int(s)) for c, s in zip(g.count(), g.sum())]


def main() -> None:
    print("Rescoring with the fallback extractor (no GPU, no cost)...")
    dfs = {kind: rescore(kind) for kind in ("base", "rl")}

    report = {"per_condition": {}, "extraction_methods": {}}
    for kind, df in dfs.items():
        unparsed = df[df["extracted_answer"].isna()]
        report["extraction_methods"][kind] = {
            k: int(v) for k, v in unparsed["method"].value_counts().items()
        }

    print()
    header = f"{'cond':<5}{'scoring':<11}{'base':>9}{'rl':>9}{'delta':>9}{'  95% CI':>22}{'  sig':>6}"
    print(header)
    print("-" * len(header))
    for condition in CONDITIONS:
        entry = {}
        for label, col in (("strict", "correct"), ("fallback", "correct_fb")):
            from src.metrics.pass_at_k import mean_pass_at_k

            b = mean_pass_at_k(per_problem(dfs["base"], condition, col), 1)
            r = mean_pass_at_k(per_problem(dfs["rl"], condition, col), 1)
            d = bootstrap_delta_ci(
                per_problem(dfs["rl"], condition, col),
                per_problem(dfs["base"], condition, col),
                k=1,
            )
            entry[label] = {
                "base_pass_at_1": round(b, 4),
                "rl_pass_at_1": round(r, 4),
                "delta": round(d["point_estimate"], 4),
                "ci": [round(d["ci_lower"], 4), round(d["ci_upper"], 4)],
                "significant": d["significant"],
            }
            ci = f"[{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
            print(f"{condition:<5}{label:<11}{b:>9.4f}{r:>9.4f}"
                  f"{d['point_estimate']:>+9.4f}{ci:>22}{str(d['significant']):>6}")
        entry["format_component"] = round(
            entry["strict"]["delta"] - entry["fallback"]["delta"], 4
        )
        report["per_condition"][condition] = entry

    Path("eval_results/rescore_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nWrote eval_results/rescore_comparison.json")
    print(
        "\nREMINDER: the permissive numbers are not publishable until "
        "scripts/build_extraction_review_pack.py has been completed and scored."
    )


if __name__ == "__main__":
    main()
