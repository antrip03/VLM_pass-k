"""
Spurious-correctness sampling (PLAN.md Section 4 item 9, Section 5 item
10): "~30-50 manually reviewed 'correct' traces, checking for
right-answer/wrong-reasoning".

NOTE ON NAMING - two different failure modes share a confusable name, and
conflating them would leave a real gap in the controls:

  * SPURIOUS CORRECTNESS (this module): the model reports the right final
    number via wrong or absent reasoning - guessing, arithmetic that
    happens to cancel, restating a number from the problem statement.
    Threatens the claim that the measured accuracy reflects reasoning.

  * SPURIOUS REWARDS (src/training/reward_fn_random.py and the control
    built on it): the training gain appears even when the reward signal
    carries no information about correctness (Shao et al., arXiv
    2506.10947). Threatens the claim that RL taught anything at all.

PLAN.md Section 4 item 9 covers only the first. The second is a separate
control and is NOT a duplicate of this one.

WHY SAMPLING NEEDS TO BE STRATIFIED AND BLINDED
-----------------------------------------------
The question is not "does spurious correctness occur" (it does, in every
model) but "does it occur at a DIFFERENT RATE in the RL model than the
base model" - because only a differential rate can manufacture a
Delta. That has two consequences for how traces are drawn:

  1. Stratify by model. An unstratified sample of "correct traces" tells
     you the pooled rate, which cannot answer the differential question.

  2. Blind the reviewer. If the reviewer knows which model produced a
     trace, the judgement is no longer independent of the hypothesis -
     and this project's whole design rests on a comparison between those
     two models. This module therefore emits traces in a SHUFFLED order
     with model identity held in a separate key file, so the review can
     be scored honestly and only then joined back.

This module does the sampling and the blinding. The judging is manual -
deliberately, since an LLM judge would introduce exactly the kind of
unvalidated measurement instrument that the extraction bug (commit
43279ab) already demonstrated this project is vulnerable to.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def sample_correct_traces(
    df,
    condition: str,
    n_per_model: int = 25,
    seed: int = 0,
) -> list[dict]:
    """
    Draw `n_per_model` correct traces from each model for one condition.

    Sampling is WITHOUT replacement at the (problem, sample) level and is
    spread across distinct problems where possible: drawing 25 traces
    that all come from two easy problems would measure those problems,
    not the model. Implemented by shuffling problems first, then taking
    at most one trace per problem until the quota is filled, wrapping
    round if there are fewer eligible problems than the quota.
    """
    rng = random.Random(seed)
    drawn: list[dict] = []

    for kind in ("base", "rl"):
        sub = df[
            (df["condition"] == condition)
            & (df["model_kind"] == kind)
            & (df["correct"])
        ]
        if len(sub) == 0:
            continue

        by_problem: dict[int, list] = {}
        for _, row in sub.iterrows():
            by_problem.setdefault(int(row["problem_idx"]), []).append(row)

        problems = list(by_problem)
        rng.shuffle(problems)
        for pool in by_problem.values():
            rng.shuffle(pool)

        picked, cursor = 0, 0
        while picked < n_per_model and problems:
            exhausted = True
            for pidx in problems:
                if cursor < len(by_problem[pidx]):
                    exhausted = False
                    row = by_problem[pidx][cursor]
                    drawn.append(
                        {
                            "model_kind": kind,
                            "condition": condition,
                            "problem_idx": int(row["problem_idx"]),
                            "sample_idx": int(row["sample_idx"]),
                            "question": str(row["question"]),
                            "ground_truth": str(row["ground_truth"]),
                            "extracted_answer": str(row["extracted_answer"]),
                            "completion": str(row["completion"]),
                        }
                    )
                    picked += 1
                    if picked >= n_per_model:
                        break
            if exhausted:
                break
            cursor += 1

    return drawn


def write_blinded_review_pack(
    traces: list[dict],
    out_dir: str | Path,
    seed: int = 0,
) -> dict:
    """
    Write two files:

      spurious_correctness_review.jsonl - shuffled traces, each with a
        `review_id` and NO model_kind field. This is what the human reads.
      spurious_correctness_key.json - review_id -> model_kind mapping.
        Do not open until every trace has been judged.

    Each review row carries a `verdict` field pre-set to null; the
    reviewer fills in one of "sound", "spurious", or "unclear".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    shuffled = list(traces)
    rng.shuffle(shuffled)

    key = {}
    review_path = out_dir / "spurious_correctness_review.jsonl"
    with open(review_path, "w", encoding="utf-8") as f:
        for i, t in enumerate(shuffled):
            review_id = f"R{i:04d}"
            key[review_id] = t["model_kind"]
            row = {
                "review_id": review_id,
                "condition": t["condition"],
                "problem_idx": t["problem_idx"],
                "question": t["question"],
                "ground_truth": t["ground_truth"],
                "extracted_answer": t["extracted_answer"],
                "completion": t["completion"],
                # Reviewer fills this in:
                #   "sound"    - reasoning genuinely derives the answer
                #   "spurious" - right number, reasoning absent/wrong/circular
                #   "unclear"  - cannot tell
                "verdict": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    key_path = out_dir / "spurious_correctness_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)

    return {
        "review_path": str(review_path),
        "key_path": str(key_path),
        "n_traces": len(shuffled),
        "instructions": (
            "Fill in `verdict` for every row in the .jsonl "
            "(sound|spurious|unclear), THEN run score_blinded_review() to "
            "join against the key. Do not open the key file first."
        ),
    }


def score_blinded_review(review_path: str | Path, key_path: str | Path) -> dict:
    """
    Join a completed review against the key and report the spurious rate
    PER MODEL - the differential that actually matters.

    Refuses to score if any verdict is still null, rather than silently
    treating unreviewed rows as sound (which would bias the rate toward
    zero exactly in proportion to how much of the review was skipped).
    """
    rows = []
    with open(review_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    with open(key_path, encoding="utf-8") as f:
        key = json.load(f)

    unreviewed = [r["review_id"] for r in rows if r.get("verdict") is None]
    if unreviewed:
        raise ValueError(
            f"{len(unreviewed)} trace(s) still unreviewed (e.g. {unreviewed[:5]}). "
            f"Scoring now would understate the spurious rate."
        )

    valid = {"sound", "spurious", "unclear"}
    bad = {r["review_id"]: r["verdict"] for r in rows if r["verdict"] not in valid}
    if bad:
        raise ValueError(f"Invalid verdicts (must be one of {sorted(valid)}): {bad}")

    out = {}
    for kind in ("base", "rl"):
        mine = [r for r in rows if key.get(r["review_id"]) == kind]
        if not mine:
            continue
        counts = {v: sum(1 for r in mine if r["verdict"] == v) for v in sorted(valid)}
        decided = counts["sound"] + counts["spurious"]
        out[kind] = {
            "n_reviewed": len(mine),
            **counts,
            # Rate over DECIDED traces only; "unclear" is reported
            # separately rather than folded silently into either bucket.
            "spurious_rate_of_decided": round(counts["spurious"] / decided, 4) if decided else None,
            "unclear_fraction": round(counts["unclear"] / len(mine), 4),
        }

    if "base" in out and "rl" in out:
        rb, rr = out["base"]["spurious_rate_of_decided"], out["rl"]["spurious_rate_of_decided"]
        if rb is not None and rr is not None:
            out["differential_rl_minus_base"] = round(rr - rb, 4)
            out["interpretation"] = (
                "A positive differential means the RL model is MORE often spuriously correct, "
                "i.e. part of the measured Delta is not reasoning. A differential near zero "
                "means spurious correctness is a shared baseline property and does not "
                "manufacture the Delta."
            )
    return out
