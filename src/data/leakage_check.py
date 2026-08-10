"""
Leakage and pairing-integrity checks (PLAN.md Section 3 Step 2; supports
the calibration requirements in Section 4).

Two distinct checks, bundled here because both answer the same underlying
question - "are we actually comparing what we think we're comparing?":

  1. check_train_eval_leakage: no exact-text overlap between whatever
     subset of GSM8K's train split is used for RL training and whatever
     subset of the test split is used for evaluation. GSM8K's own HF
     train/test split should already guarantee this; this is a defensive
     check, not a decorative one - datasets occasionally have accidental
     near-duplicates across splits.

  2. check_text_image_pairing: confirms each eval problem's rendered
     image genuinely corresponds to that exact problem's text - guards
     against an off-by-one or shuffled-pairing bug silently comparing the
     wrong content across the T/D/E conditions, which would quietly
     invalidate every downstream Delta_text/Delta_pixel number.
"""

from __future__ import annotations

from .gsm8k_loader import GSM8KExample


def check_train_eval_leakage(
    train_examples: list[GSM8KExample], eval_examples: list[GSM8KExample]
) -> dict:
    """
    Returns {"overlap_count": int, "overlap_questions": [...], ...}.
    overlap_count must be 0 before trusting any downstream result -
    Phase 0/1 orchestration (Step 8) should treat overlap_count > 0 as a
    hard stop, not a warning.
    """
    train_questions = {ex.question.strip().lower() for ex in train_examples}
    overlap = [
        ex.question
        for ex in eval_examples
        if ex.question.strip().lower() in train_questions
    ]
    return {
        "overlap_count": len(overlap),
        "overlap_questions": overlap,
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
    }


def check_text_image_pairing(
    examples: list[GSM8KExample], rendered_source_texts: dict[int, str]
) -> dict:
    """
    `rendered_source_texts` maps example.idx -> the exact text string that
    was actually passed into render_problem_image() for that example.
    Confirms it matches example.question exactly (byte-for-byte, after the
    same .strip() normalization the loader already applies).
    """
    mismatches = [
        ex.idx
        for ex in examples
        if rendered_source_texts.get(ex.idx) != ex.question
    ]
    return {
        "mismatch_count": len(mismatches),
        "mismatch_indices": mismatches,
        "total_checked": len(examples),
    }
