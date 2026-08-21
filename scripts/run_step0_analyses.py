"""
Step 0 of the Phase 1b control plan: every analysis that costs NOTHING
because it reads the evaluation records already produced by
scripts/run_phase1_eval_on_modal.py.

Runs LOCALLY on the two saved parquet files - no Modal, no GPU, no
credentials. That is deliberate: these analyses are pure functions of
records that already exist, and making them require cloud compute would
put a price on the cheapest, highest-value work in the plan.

    python3 scripts/run_step0_analyses.py \
        --base records_base.parquet --rl records_rl.parquet --out step0_report.json

WHAT IT PRODUCES, AND WHICH REVIEWER OBJECTION EACH ONE ANSWERS
---------------------------------------------------------------
1. COVERAGE (src/analysis/coverage.py)
   Objection: "your pass@64 is at 0.98, so 'no expansion' is
   indistinguishable from 'no room'."
   Answer: per-problem reachability and within-problem density shift,
   neither of which depends on where the aggregate ceiling sits.

2. FORMAT CONFOUND (src/analysis/format_confound.py)
   Objection: "commit 43279ab showed 40% of Delta_text was format
   compliance; PLAN.md says a residual asymmetry remains - how do you
   know the rest isn't too?"
   Answer: a tipping-point sensitivity analysis plus a measured upper
   bound on hidden correctness.

3. TRUNCATION SYMMETRY (src/controls/truncation_check.py)
   Objection: "the 1200-token cap could be clipping one model more."
   PLAN.md Section 3 Phase 1 item 7 calls this mandatory.

4. TRANSCRIPTION FIDELITY (src/metrics/transcription_fidelity.py)
   Objection: "condition D's gap could be perception, not reasoning."
   PLAN.md Section 4 item 4.

5. SPURIOUS-CORRECTNESS REVIEW PACK (src/controls/spurious_correctness.py)
   Objection: "right answer, wrong reasoning." PLAN.md Section 4 item 9.
   Emits a BLINDED pack for manual review; it does not score anything by
   itself.

A NOTE ON WHAT THIS SCRIPT DOES NOT DO
--------------------------------------
It does not recompute pass@k or the headline Deltas - those already exist
in analysis.json from the eval run, and recomputing them here from a
second code path would invite the two to silently disagree. This script
adds only what analysis.json does not already contain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.analysis.coverage import full_coverage_report
from src.analysis.format_confound import full_format_confound_report
from src.controls.difficulty_matched import full_difficulty_report
from src.controls.spurious_correctness import sample_correct_traces, write_blinded_review_pack

EXPECTED_COLUMNS = {
    "problem_idx",
    "question",
    "ground_truth",
    "condition",
    "sample_idx",
    "completion",
    "extracted_answer",
    "transcription",
    "correct",
}


def load_paired_records(base_path: str, rl_path: str) -> pd.DataFrame:
    """
    Load both models' records into one frame with a `model_kind` column.

    Validates the schema up front rather than letting a missing column
    surface as a confusing KeyError three analyses deep - these files may
    arrive from a different machine or a re-run, and a silent schema drift
    is exactly the kind of thing that produced the checkpoint-loading trap
    documented in run_phase1_eval_on_modal.py.
    """
    frames = []
    for kind, path in (("base", base_path), ("rl", rl_path)):
        if not Path(path).exists():
            raise SystemExit(f"Missing records file for {kind!r}: {path}")
        df = pd.read_parquet(path)
        missing = EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise SystemExit(f"{path} is missing expected columns: {sorted(missing)}")
        df = df.copy()
        df["model_kind"] = kind
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Sanity: the two models must cover the same problems, or every
    # "paired" statistic downstream is quietly comparing different sets.
    per_kind = {k: set(g["problem_idx"]) for k, g in combined.groupby("model_kind")}
    if per_kind["base"] != per_kind["rl"]:
        only_base = sorted(per_kind["base"] - per_kind["rl"])
        only_rl = sorted(per_kind["rl"] - per_kind["base"])
        raise SystemExit(
            f"Problem sets differ between models - base-only={only_base[:10]}, "
            f"rl-only={only_rl[:10]}. Paired analyses would be invalid."
        )
    return combined


def truncation_report(df, max_new_tokens: int) -> dict:
    """
    Truncation symmetry from the saved records.

    IMPORTANT LIMITATION, stated rather than papered over: the parquet
    records store decoded TEXT, not the output token ids, so
    src/controls/truncation_check.py's hit_max_tokens() - which needs the
    ids and the EOS id - cannot be applied here. What we can measure is
    the completion-length distribution and the rate of long completions
    that also failed to parse, which is the observable SYMPTOM of
    truncation that PLAN.md item 7 actually asks to be compared between
    models. A length distribution bunched hard against the cap in one
    model and not the other is the signal; this reports the inputs for
    that judgement rather than asserting a binary verdict it cannot
    support from this data.
    """
    out = {"max_new_tokens_configured": max_new_tokens, "note": truncation_report.__doc__.strip()}
    for condition in ("T", "D", "E"):
        entry = {}
        for kind in ("base", "rl"):
            sub = df[(df["condition"] == condition) & (df["model_kind"] == kind)]
            if len(sub) == 0:
                continue
            lengths = sub["completion"].astype(str).str.len()
            unparsed_long = sub[
                sub["extracted_answer"].isna() & (lengths > lengths.quantile(0.9))
            ]
            entry[kind] = {
                "n": len(sub),
                "chars_mean": round(float(lengths.mean()), 1),
                "chars_p50": int(lengths.quantile(0.5)),
                "chars_p90": int(lengths.quantile(0.9)),
                "chars_p99": int(lengths.quantile(0.99)),
                "chars_max": int(lengths.max()),
                "unparsed_among_longest_decile": len(unparsed_long),
            }
        if "base" in entry and "rl" in entry:
            entry["p99_gap_rl_minus_base"] = entry["rl"]["chars_p99"] - entry["base"]["chars_p99"]
        out[condition] = entry
    return out


def transcription_report(df) -> dict:
    """Condition-D transcription fidelity, per model (PLAN.md Section 4 item 4)."""
    from src.metrics.transcription_fidelity import mean_transcription_fidelity

    out = {}
    for kind in ("base", "rl"):
        sub = df[
            (df["condition"] == "D")
            & (df["model_kind"] == kind)
            & df["transcription"].notna()
        ]
        if len(sub) == 0:
            out[kind] = {"note": "no D-condition transcriptions found in records"}
            continue
        pairs = [(str(r["transcription"]), str(r["question"])) for _, r in sub.iterrows()]
        out[kind] = {"n_transcriptions": len(pairs), **mean_transcription_fidelity(pairs)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1b Step 0: free analyses on existing eval records.")
    ap.add_argument("--base", required=True, help="records_base.parquet")
    ap.add_argument("--rl", required=True, help="records_rl.parquet")
    ap.add_argument("--out", default="step0_report.json")
    ap.add_argument("--review-dir", default="review_pack")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--traces-per-model", type=int, default=25)
    args = ap.parse_args()

    df = load_paired_records(args.base, args.rl)
    print(f"Loaded {len(df):,} records "
          f"({df['problem_idx'].nunique()} problems x {sorted(df['condition'].unique())} "
          f"x {sorted(df['model_kind'].unique())})")

    report = {
        "record_counts": {
            f"{k}/{c}": int(((df.model_kind == k) & (df.condition == c)).sum())
            for k in ("base", "rl")
            for c in ("T", "D", "E")
        },
        "coverage": full_coverage_report(df),
        "format_confound": full_format_confound_report(df),
        # The difficulty-matched control was originally scoped as a paid
        # step (screen a larger pool, then evaluate a matched subset).
        # The STRATIFIED form needs no new generation at all: T, D and E
        # were evaluated on the SAME problems, so binning by base
        # difficulty and comparing Delta within bins tests the
        # floor-effect explanation using records that already exist.
        # Only if this comes back inconclusive is the paid version worth
        # running.
        "difficulty_matched": full_difficulty_report(df),
        "truncation": truncation_report(df, args.max_new_tokens),
        "transcription_fidelity": transcription_report(df),
    }

    traces = []
    for condition in ("T", "D", "E"):
        traces.extend(sample_correct_traces(df, condition, n_per_model=args.traces_per_model))
    report["spurious_correctness_pack"] = write_blinded_review_pack(traces, args.review_dir)

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    # Print the two findings that most affect what the paper can claim.
    print("\n=== COVERAGE (ceiling-independent) ===")
    for cond, r in report["coverage"].items():
        t1 = r["threshold_sweep"][0]
        d = r["density_shift"]
        flag = "  [UNDERPOWERED]" if t1["underpowered"] else ""
        print(f"  {cond}: expansion={t1['expansion_count']} contraction={t1['contraction_count']} "
              f"(McNemar p={t1['mcnemar_p_exact']}){flag}")
        print(f"      within-coverage density shift: {d.get('mean_density_shift')} "
              f"(up={d.get('n_rate_increased')} down={d.get('n_rate_decreased')})")

    print("\n=== RESIDUAL FORMAT CONFOUND ===")
    for cond, r in report["format_confound"]["per_condition"].items():
        print(f"  {cond}: {r['verdict'] or r['sensitivity']['verdict']}")

    print("\n=== FLOOR EFFECT (is Delta_pixel > Delta_text real?) ===")
    for key in ("stratified_T_vs_E", "stratified_T_vs_D"):
        v = report["difficulty_matched"].get(key, {})
        print(f"  {key}: {v.get('verdict') or v.get('note')}")


if __name__ == "__main__":
    main()
