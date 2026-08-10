"""
GSM8K text data loading (PLAN.md Section 3, Step 2).

Loads the standard openai/gsm8k "main" config train/test splits and
extracts the ground-truth final numeric answer from each example's
`answer` field (format: reasoning steps with <<calculator>> annotations,
ending in "#### <number>").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


@dataclass(frozen=True)
class GSM8KExample:
    """One GSM8K problem: the question text plus its verified final answer."""

    idx: int
    split: str  # "train" or "test"
    question: str
    answer_text: str  # full original `answer` field, reasoning + "#### N"
    final_answer: str  # extracted, normalized numeric answer, e.g. "72"


def extract_final_answer(answer_field: str) -> str:
    """
    Extract and normalize the ground-truth numeric answer from GSM8K's
    `answer` field (e.g. "...#### 72" -> "72"). Normalizes by stripping
    thousands-separator commas and surrounding whitespace.

    Raises ValueError if the "#### <number>" marker isn't found - fail
    loudly rather than silently returning something wrong if the dataset
    format ever changes upstream.
    """
    match = _ANSWER_RE.search(answer_field)
    if match is None:
        raise ValueError(
            f"Could not find '#### <number>' pattern in answer field: {answer_field!r}"
        )
    return match.group(1).replace(",", "").strip()


def load_gsm8k(split: str) -> list[GSM8KExample]:
    """
    Load a GSM8K split ("train" or "test") and return a list of
    GSM8KExample with the ground-truth final answer pre-extracted.

    "train" is the pool used for text-only RL training (PLAN.md Phase 1);
    "test" is the pool used for evaluation (pass@k across T/D/E
    conditions) - these are two different, disjoint HF splits, checked
    for accidental overlap by leakage_check.check_train_eval_leakage.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples = []
    for idx, row in enumerate(ds):
        examples.append(
            GSM8KExample(
                idx=idx,
                split=split,
                question=row["question"].strip(),
                answer_text=row["answer"],
                final_answer=extract_final_answer(row["answer"]),
            )
        )
    return examples
