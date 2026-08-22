"""
Build a manual review pack to VALIDATE the fallback extractor
(src/metrics/answer_extraction_fallback.py) before any number derived
from it goes in the paper.

WHY THIS IS MANDATORY
---------------------
Rescoring the committed Phase 1 records with the fallback extractor moves
every headline delta to ~zero:

    Delta_T  +0.0587 [+0.0406,+0.0763]  ->  -0.0081 [-0.0212,+0.0042]
    Delta_D  +0.0719 [+0.0558,+0.0889]  ->  +0.0019 [-0.0109,+0.0150]
    Delta_E  +0.0925 [+0.0695,+0.1161]  ->  -0.0061 [-0.0209,+0.0092]

That is a paper-changing result, and it rests on an extractor. This
project has already been burned once by an unvalidated extractor
producing a plausible but wrong headline (commit 43279ab, in the opposite
direction). The same standard must apply to a change that cuts the other
way: an extractor that erases a result is exactly as suspect as one that
manufactures one, and deserves exactly as much scrutiny.

WHAT IS ALREADY KNOWN WITHOUT MANUAL WORK
-----------------------------------------
Two automated checks already support the fallback (2026-08-22):

  1. HELD-OUT RECOVERY. On completions that DID emit "####", deleting the
     marker line and re-extracting recovers the same answer 84.3%
     (base) / 84.5% (rl) of the time, wrong 14.5% / 14.8%. The error rate
     is SYMMETRIC between models, so it does not bias the contrast, and
     most wrong extractions will not coincidentally equal the ground
     truth, so the fallback tends to UNDERSTATE accuracy.

  2. POPULATION DIAGNOSTIC. Base's unparsed completions score about the
     same as its parsed ones (gaps -0.048 / -0.008 / +0.009 on T/D/E),
     i.e. failing to emit "####" was uncorrelated with being wrong. RL's
     unparsed completions score far worse (-0.193 / -0.117 / -0.089),
     which is what you would expect when a model trained into compliance
     fails to comply - those are degenerate generations, a different
     population.

Manual review is the remaining step: those checks measure the extractor
against ITSELF and against marker-emitting completions, not against a
human judgement of what the model actually meant on the no-marker
population that matters.

WHAT THE REVIEWER DOES
----------------------
For each sampled completion, read the tail and record:

  true_answer : the number the model actually concluded with, as a human
                reads it - or "none" if it never states a final answer.
  reasoning   : "sound" | "wrong" | "unclear" - whether the derivation
                actually supports that number.

The extractor's own output is deliberately WITHHELD from the review file
(it is stored in the key), so the reviewer is not anchored by it. Scoring
then measures precision: how often the extractor's answer equals the
human's true_answer.

    python3 scripts/build_extraction_review_pack.py --n 60
    # ... fill in extraction_review.jsonl ...
    python3 scripts/build_extraction_review_pack.py --score
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.metrics.answer_extraction_fallback import extract_with_fallback, normalize_number

DEFAULT_DIR = Path("eval_results/extraction_review")


def build(n_per_model: int, conditions: tuple[str, ...], out_dir: Path, seed: int) -> dict:
    rng = random.Random(seed)
    rows, key = [], {}

    for kind in ("base", "rl"):
        df = pd.read_parquet(f"eval_results/records_{kind}.parquet")
        pool = df[df["condition"].isin(conditions) & df["extracted_answer"].isna()]
        if len(pool) == 0:
            continue
        take = pool.sample(min(n_per_model, len(pool)), random_state=seed)
        for _, r in take.iterrows():
            completion = str(r["completion"])
            predicted, method = extract_with_fallback(completion)
            rows.append(
                {
                    "model_kind": kind,
                    "condition": str(r["condition"]),
                    "problem_idx": int(r["problem_idx"]),
                    "ground_truth": str(r["ground_truth"]),
                    "completion_tail": completion[-700:],
                    "_predicted": predicted,
                    "_method": method,
                }
            )

    rng.shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_path = out_dir / "extraction_review.jsonl"
    with open(review_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            rid = f"X{i:04d}"
            key[rid] = {
                "model_kind": row["model_kind"],
                "predicted": row["_predicted"],
                "method": row["_method"],
                "ground_truth": row["ground_truth"],
            }
            f.write(
                json.dumps(
                    {
                        "review_id": rid,
                        "condition": row["condition"],
                        "completion_tail": row["completion_tail"],
                        # Reviewer fills these two in. The extractor's
                        # answer is NOT shown, to avoid anchoring.
                        "true_answer": None,
                        "reasoning": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    key_path = out_dir / "extraction_review_key.json"
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return {
        "review_path": str(review_path),
        "key_path": str(key_path),
        "n": len(rows),
        "instructions": (
            "For each row set true_answer (the number the model concluded with, or "
            "\"none\") and reasoning (sound|wrong|unclear). Ground truth is NOT shown - "
            "you are recording what the model SAID, not whether it was right. "
            "Then re-run with --score."
        ),
    }


def score(out_dir: Path) -> dict:
    review_path, key_path = out_dir / "extraction_review.jsonl", out_dir / "extraction_review_key.json"
    rows = [json.loads(l) for l in review_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    key = json.loads(key_path.read_text(encoding="utf-8"))

    missing = [r["review_id"] for r in rows if r.get("true_answer") is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} rows unreviewed (e.g. {missing[:5]}). Scoring an incomplete "
            f"review would bias precision toward whichever rows were easiest to judge."
        )

    out = {}
    for kind in ("base", "rl"):
        mine = [r for r in rows if key[r["review_id"]]["model_kind"] == kind]
        if not mine:
            continue
        hit = miss = human_none = 0
        for r in mine:
            human = str(r["true_answer"]).strip().lower()
            pred = key[r["review_id"]]["predicted"]
            if human in ("none", "null", ""):
                human_none += 1
                continue
            if pred is not None and normalize_number(str(pred)) == normalize_number(human):
                hit += 1
            else:
                miss += 1
        judged = hit + miss
        out[kind] = {
            "n_reviewed": len(mine),
            "human_says_no_answer": human_none,
            "extractor_correct": hit,
            "extractor_wrong": miss,
            "precision": round(hit / judged, 4) if judged else None,
            "reasoning_sound": sum(1 for r in mine if r.get("reasoning") == "sound"),
            "reasoning_wrong": sum(1 for r in mine if r.get("reasoning") == "wrong"),
            "reasoning_unclear": sum(1 for r in mine if r.get("reasoning") == "unclear"),
        }

    if "base" in out and "rl" in out and out["base"]["precision"] and out["rl"]["precision"]:
        gap = out["rl"]["precision"] - out["base"]["precision"]
        out["precision_gap_rl_minus_base"] = round(gap, 4)
        out["interpretation"] = (
            "Precision must be similar across models. A large gap would mean the fallback "
            "extractor treats one model's output more favourably, which would invalidate the "
            "rescored contrast rather than merely add noise. A symmetric error rate only adds "
            "noise and, since a wrong extraction rarely equals the ground truth, biases the "
            "rescored accuracy DOWNWARD for both models."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="samples per model")
    ap.add_argument("--conditions", default="T,E")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    if args.score:
        print(json.dumps(score(out_dir), indent=2))
    else:
        conds = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
        print(json.dumps(build(args.n, conds, out_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
