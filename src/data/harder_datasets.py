"""
Harder / perturbed evaluation sets, for the two Phase 1b problems that
GSM8K alone cannot solve.

PROBLEM 1 - THE CEILING (Steps 2-3)
-----------------------------------
PLAN.md's own Phase 1 result box states the limitation plainly: base
pass@64 is 0.976-0.990, so "there is almost no room to detect coverage
expansion, so 'no expansion' and 'cannot tell' are hard to separate
here." Both pass@64 deltas are in fact non-significant. The sharpening
claim - Finding 2, half the paper - is measured at saturation.

Fixing this needs a harder evaluation set, NOT retraining: both
checkpoints already exist, so this is eval-only cost.

PROBLEM 2 - THE PARAPHRASE CONTROL (Step 6)
-------------------------------------------
src/data/paraphrase_ood.py is a stub. Its own docstring is honest about
it: it swaps one proper noun for another from a fixed pool and "does not
meaningfully vary phrasing beyond that". Renaming Janet to Maria is not a
distribution shift, and a null result from it would be meaningless.

BOTH ARE SOLVED BY GSM-PLUS
---------------------------
GSM-Plus (Li et al., ACL 2024, arXiv 2402.19255; HF `qintongli/GSM-Plus`)
is GSM8K perturbed along eight axes, with the answer recomputed for each
perturbation. Two of those axes are exactly what we need:

  * "problem understanding" = REPHRASING with the same underlying maths.
    That is the paraphrase-OOD control, done by a peer-reviewed source
    rather than by us - so we are not the ones vouching that the answer
    survived the rewrite.

  * "adding operation" / "reversing operation" / "distraction insertion"
    / "digit expansion" = genuinely HARDER variants of the same problems.
    That is the ceiling fix.

Crucially it also ships `seed_question`, the original GSM8K item each
perturbation derives from. That lets the comparison be PAIRED - same
underlying problem, original vs perturbed - which at 50 problems is a
materially tighter contrast than two independent samples.

ONE PERTURBATION TYPE IS DELIBERATELY EXCLUDED
----------------------------------------------
"critical thinking" perturbs problems by REMOVING a necessary condition,
so the correct response is to say the problem is unanswerable rather than
to produce a number. This project's entire scoring path is numeric exact
match (src/metrics/answer_extraction.py), so those items would be scored
as failures for both models no matter what they output - adding noise,
not difficulty. Excluded by default, and the exclusion is explicit rather
than incidental.

ON MATH-500
-----------
MATH-500 is the other obvious harder set and is what Shao et al. use. It
is supported here but NOT the default, for a concrete reason: MATH
answers are frequently LaTeX expressions (\\frac{1}{2}, \\sqrt{3},
intervals), and this project scores by numeric exact match. Running it
unfiltered would score correct symbolic answers as wrong and produce a
second measurement artifact of exactly the kind commit 43279ab already
cost this project once. `load_math500(numeric_only=True)` filters to
items whose reference answer parses as a plain number; the fraction
retained is reported so the filtering is visible rather than silent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.data.gsm8k_loader import GSM8KExample

GSM_PLUS_REPO = "qintongli/GSM-Plus"
MATH500_REPO = "HuggingFaceH4/MATH-500"

# Verified against the dataset card (2026-08-22). Exact strings matter -
# a typo here silently yields an empty set rather than an error.
PERTURBATION_REPHRASE = "problem understanding"
PERTURBATION_HARDER = (
    "adding operation",
    "reversing operation",
    "distraction insertion",
    "digit expansion",
)
PERTURBATION_EXCLUDED = ("critical thinking",)

_NUMERIC_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?$")


@dataclass(frozen=True)
class PerturbedPair:
    """
    One GSM8K item in both its original and perturbed form.

    `original` and `perturbed` share the same `idx`, so any downstream
    per-problem pairing (bootstrap_delta_ci, coverage analysis) lines them
    up automatically without a separate join.
    """

    idx: int
    perturbation_type: str
    original: GSM8KExample
    perturbed: GSM8KExample


def _normalize_numeric(value) -> str | None:
    """
    Normalise a reference answer to this project's canonical numeric
    string, or return None if it is not a plain number.

    Returning None rather than raising is deliberate: upstream datasets
    legitimately contain non-numeric answers, and the caller's job is to
    filter and REPORT how many were dropped, not to crash on the first
    one.
    """
    if value is None:
        return None
    s = str(value).strip()
    # Strip common decoration that carries no numeric meaning.
    s = s.replace("$", "").replace("\\", "").replace("%", "").strip()
    s = s.rstrip(".")
    if not _NUMERIC_RE.match(s):
        return None
    s = s.replace(",", "")
    # Canonicalise "18.0" -> "18" so it compares equal to a model that
    # writes the integer, matching GSM8K's own integer-style answers.
    if "." in s:
        try:
            f = float(s)
            if f == int(f):
                s = str(int(f))
        except ValueError:
            return None
    return s


def load_gsm_plus_pairs(
    perturbation_types: tuple[str, ...],
    limit: int | None = None,
    split: str = "test",
    seed: int = 0,
) -> tuple[list[PerturbedPair], dict]:
    """
    Load paired (original, perturbed) items for the requested
    perturbation types.

    Returns (pairs, stats). `stats` reports how many rows were seen,
    dropped for a non-numeric answer, and retained - so any filtering is
    a reported number in the paper rather than an invisible decision.

    Selection is deterministic: rows are sorted by (perturbation_type,
    seed_question) then shuffled with a fixed seed before truncation, so
    `limit` takes a reproducible, type-balanced sample rather than
    whatever order the Hub happens to return.
    """
    import random

    from datasets import load_dataset

    for t in perturbation_types:
        if t in PERTURBATION_EXCLUDED:
            raise ValueError(
                f"Perturbation type {t!r} is excluded by design - see this module's docstring "
                f"(its correct answer is 'unanswerable', which numeric exact-match scoring "
                f"cannot represent)."
            )

    ds = load_dataset(GSM_PLUS_REPO, split=split)
    seen = dropped_non_numeric = 0
    rows = []
    for row in ds:
        if row.get("perturbation_type") not in perturbation_types:
            continue
        seen += 1
        pert_ans = _normalize_numeric(row.get("answer"))
        seed_ans = _normalize_numeric(row.get("seed_answer"))
        if pert_ans is None or seed_ans is None:
            dropped_non_numeric += 1
            continue
        rows.append((row["perturbation_type"], row["seed_question"], row["question"], seed_ans, pert_ans))

    rows.sort(key=lambda r: (r[0], r[1]))
    random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]

    pairs = []
    for idx, (ptype, seed_q, pert_q, seed_ans, pert_ans) in enumerate(rows):
        pairs.append(
            PerturbedPair(
                idx=idx,
                perturbation_type=ptype,
                original=GSM8KExample(
                    idx=idx,
                    split=f"gsmplus-{split}-original",
                    question=seed_q.strip(),
                    answer_text=f"#### {seed_ans}",
                    final_answer=seed_ans,
                ),
                perturbed=GSM8KExample(
                    idx=idx,
                    split=f"gsmplus-{split}-{ptype.replace(' ', '_')}",
                    question=pert_q.strip(),
                    answer_text=f"#### {pert_ans}",
                    final_answer=pert_ans,
                ),
            )
        )

    stats = {
        "repo": GSM_PLUS_REPO,
        "split": split,
        "perturbation_types": list(perturbation_types),
        "rows_matching_type": seen,
        "dropped_non_numeric_answer": dropped_non_numeric,
        "pairs_returned": len(pairs),
        "limit": limit,
        "seed": seed,
        "type_breakdown": {
            t: sum(1 for p in pairs if p.perturbation_type == t) for t in perturbation_types
        },
    }
    return pairs, stats


def load_gsm_plus_harder(limit: int | None = None, **kw) -> tuple[list[PerturbedPair], dict]:
    """The ceiling-fix set: harder perturbations, paired with their originals."""
    return load_gsm_plus_pairs(PERTURBATION_HARDER, limit=limit, **kw)


def load_gsm_plus_rephrased(limit: int | None = None, **kw) -> tuple[list[PerturbedPair], dict]:
    """The paraphrase-OOD control set: same maths, reworded, paired with originals."""
    return load_gsm_plus_pairs((PERTURBATION_REPHRASE,), limit=limit, **kw)


def load_math500(
    limit: int | None = None,
    numeric_only: bool = True,
    levels: tuple[int, ...] | None = None,
    seed: int = 0,
) -> tuple[list[GSM8KExample], dict]:
    """
    MATH-500, optionally filtered to numerically-scorable items.

    `levels` selects MATH difficulty levels (1 easiest - 5 hardest). Note
    the trade-off explicitly: the hardest levels break the ceiling most
    effectively but also carry the longest solutions, and this project's
    generation cap is 1200 tokens. Asymmetric truncation between base and
    RL would be a fresh confound, so whichever level mix is chosen must be
    run through the truncation comparison in Step 0 before its numbers
    are trusted.
    """
    import random

    from datasets import load_dataset

    ds = load_dataset(MATH500_REPO, split="test")
    seen = dropped = 0
    rows = []
    for row in ds:
        if levels is not None and int(row.get("level", 0)) not in levels:
            continue
        seen += 1
        ans = _normalize_numeric(row.get("answer"))
        if numeric_only and ans is None:
            dropped += 1
            continue
        rows.append((row["problem"], ans if ans is not None else str(row.get("answer")), row.get("level"),
                     row.get("subject")))

    rows.sort(key=lambda r: r[0])
    random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]

    examples = [
        GSM8KExample(
            idx=idx,
            split="math500",
            question=problem.strip(),
            answer_text=f"#### {answer}",
            final_answer=answer,
        )
        for idx, (problem, answer, _lvl, _subj) in enumerate(rows)
    ]
    stats = {
        "repo": MATH500_REPO,
        "levels": list(levels) if levels else "all",
        "rows_seen": seen,
        "dropped_non_numeric_answer": dropped,
        "retained_fraction": round(1 - dropped / seen, 4) if seen else None,
        "examples_returned": len(examples),
        "numeric_only": numeric_only,
    }
    return examples, stats


def load_gsm_plus_for_problems(
    problems: list[GSM8KExample],
    perturbation_types: tuple[str, ...],
    split: str = "test",
    seed: int = 0,
) -> tuple[list[GSM8KExample], dict]:
    """
    Perturbed variants of a SPECIFIC set of problems, with `idx`
    preserved from the input.

    WHY THIS IS THE RIGHT WAY TO RUN BOTH CONTROLS
    ----------------------------------------------
    Verified 2026-08-22: GSM-Plus's rephrase subset covers all 1319 GSM8K
    test items, so all 50 of this project's evaluation problems have a
    variant. That makes a much stronger design available than sampling
    fresh problems:

      * The perturbed set is the SAME underlying problems as Phase 1, so
        problem selection is held constant. Any change in accuracy is
        attributable to the perturbation, not to having drawn an easier
        or harder sample.

      * `idx` is carried over from the input, so the resulting records
        pair directly against the existing records_base/records_rl
        parquets by problem_idx - which bootstrap_delta_ci and the
        coverage analysis both require, and which at 50 problems is a
        materially tighter contrast than unpaired sampling.

      * For the rephrase control the ORIGINALS DO NOT NEED RE-RUNNING.
        Phase 1 already measured them. Only the rephrased variants have
        to be generated, halving that control's cost.

    Where a problem has several eligible perturbations (the harder set has
    four types), one is chosen deterministically per problem from a
    seeded shuffle, and the chosen type is reported per problem so the
    mix is visible rather than incidental.
    """
    import random

    from datasets import load_dataset

    for t in perturbation_types:
        if t in PERTURBATION_EXCLUDED:
            raise ValueError(f"Perturbation type {t!r} is excluded by design - see module docstring.")

    ds = load_dataset(GSM_PLUS_REPO, split=split)
    by_seed_question: dict[str, list[tuple[str, str, str]]] = {}
    for row in ds:
        ptype = row.get("perturbation_type")
        if ptype not in perturbation_types:
            continue
        ans = _normalize_numeric(row.get("answer"))
        if ans is None:
            continue
        by_seed_question.setdefault(row["seed_question"].strip(), []).append(
            (ptype, row["question"].strip(), ans)
        )

    rng = random.Random(seed)
    out, chosen_types, missing = [], {}, []
    for p in problems:
        candidates = by_seed_question.get(p.question.strip())
        if not candidates:
            missing.append(p.idx)
            continue
        ptype, question, answer = rng.choice(sorted(candidates))
        chosen_types[p.idx] = ptype
        out.append(
            GSM8KExample(
                idx=p.idx,  # preserved - this is what makes the pairing work
                split=f"gsmplus-{ptype.replace(' ', '_')}",
                question=question,
                answer_text=f"#### {answer}",
                final_answer=answer,
            )
        )

    stats = {
        "repo": GSM_PLUS_REPO,
        "split": split,
        "perturbation_types": list(perturbation_types),
        "problems_requested": len(problems),
        "problems_matched": len(out),
        "problems_missing": missing,
        "coverage": round(len(out) / len(problems), 4) if problems else None,
        "type_breakdown": {
            t: sum(1 for v in chosen_types.values() if v == t) for t in perturbation_types
        },
        "chosen_type_by_problem_idx": chosen_types,
        "seed": seed,
    }
    return out, stats


def load_harder_for_problems(problems, **kw):
    """Harder perturbations of a given problem set (ceiling fix, Steps 2-3)."""
    return load_gsm_plus_for_problems(problems, PERTURBATION_HARDER, **kw)


def load_rephrased_for_problems(problems, **kw):
    """Rephrased variants of a given problem set (paraphrase-OOD control, Step 6)."""
    return load_gsm_plus_for_problems(problems, (PERTURBATION_REPHRASE,), **kw)


# Registry so scripts can name a dataset on the command line rather than
# importing a specific loader - keeps the eval scripts dataset-agnostic.
HARDER_DATASETS = {
    "gsmplus_harder": load_harder_for_problems,
    "gsmplus_rephrased": load_rephrased_for_problems,
}
