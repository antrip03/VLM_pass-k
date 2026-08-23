"""
Multi-extractor robustness: does the Phase 1b conclusion depend on the
extractor we wrote, or on the data?

Rescores the SAME saved completions with five independently-motivated
extractors (src/analysis/extractor_robustness.py) and reports Delta
pass@1 per condition under each, with paired-bootstrap CIs.

Free - no GPU, no Modal. The completions are already on disk.

    python3 scripts/run_extractor_robustness.py
    python3 scripts/run_extractor_robustness.py --tag gsmplus_harder_full \
        --base eval_results/phase1b/... --rl eval_results/phase1b/...

WHAT TO CONCLUDE
----------------
* strict large-and-significant while every permissive extractor is ~0
  -> the gain is a property of the SCORING RULE, not the model.
* permissive extractors disagreeing with each other
  -> no conclusion is safe from any of them; the manual review becomes
     mandatory rather than confirmatory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analysis.extractor_robustness import EXTRACTORS, score_with_all
from src.metrics.bootstrap_ci import bootstrap_delta_ci
from src.metrics.pass_at_k import mean_pass_at_k


def per_problem(df, condition, col):
    g = df[df["condition"] == condition].groupby("problem_idx")[col]
    return [(int(c), int(s)) for c, s in zip(g.count(), g.sum())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="eval_results/records_base.parquet")
    ap.add_argument("--rl", default="eval_results/records_rl.parquet")
    ap.add_argument("--conditions", default="E,D,T", help="D/E first - they are the contribution")
    ap.add_argument("--out", default="eval_results/extractor_robustness.json")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    dfs = {}
    for kind, path in (("base", args.base), ("rl", args.rl)):
        df = pd.read_parquet(path)
        print(f"scoring {kind}: {len(df):,} completions x {len(EXTRACTORS)} extractors ...",
              flush=True)
        df, cols = score_with_all(df)
        dfs[kind] = df
    print()

    report = {"extractors": list(EXTRACTORS), "per_condition": {}}
    header = f"{'condition':<10}{'extractor':<26}{'base':>9}{'RL':>9}{'delta':>9}{'  95% CI':>22}{'  sig':>6}"

    for condition in conditions:
        print(header)
        print("-" * len(header))
        entry = {}
        for name in EXTRACTORS:
            col = f"ok__{name}"
            b = mean_pass_at_k(per_problem(dfs["base"], condition, col), 1)
            r = mean_pass_at_k(per_problem(dfs["rl"], condition, col), 1)
            d = bootstrap_delta_ci(
                per_problem(dfs["rl"], condition, col),
                per_problem(dfs["base"], condition, col),
                k=1,
            )
            ci = f"[{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
            print(f"{condition:<10}{name:<26}{b:>9.4f}{r:>9.4f}"
                  f"{d['point_estimate']:>+9.4f}{ci:>22}{str(d['significant']):>6}")
            entry[name] = {
                "base_pass_at_1": round(b, 4),
                "rl_pass_at_1": round(r, 4),
                "delta": round(d["point_estimate"], 4),
                "ci": [round(d["ci_lower"], 4), round(d["ci_upper"], 4)],
                "significant": d["significant"],
            }

        # Verdict: does the conclusion depend on which permissive
        # extractor is used?
        permissive = [n for n in EXTRACTORS if n != "strict"]
        sigs = [entry[n]["significant"] for n in permissive]
        deltas = [entry[n]["delta"] for n in permissive]
        spread = max(deltas) - min(deltas)
        if entry["strict"]["significant"] and not any(sigs):
            verdict = (
                f"ROBUST: strict reports {entry['strict']['delta']:+.4f} (significant) while all "
                f"{len(permissive)} permissive extractors report no significant gain "
                f"(spread {spread:.4f}). The gain is a property of the scoring rule."
            )
        elif any(sigs) and not all(sigs):
            verdict = (
                f"MIXED: permissive extractors disagree ({sum(sigs)}/{len(sigs)} significant, "
                f"spread {spread:.4f}). No conclusion is safe from any single one - manual "
                f"review is required, not optional."
            )
        else:
            verdict = (
                f"strict={entry['strict']['delta']:+.4f} "
                f"(sig={entry['strict']['significant']}); permissive spread {spread:.4f}, "
                f"{sum(sigs)}/{len(sigs)} significant."
            )
        entry["_verdict"] = verdict
        print(f"  -> {verdict}\n")
        report["per_condition"][condition] = entry

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
