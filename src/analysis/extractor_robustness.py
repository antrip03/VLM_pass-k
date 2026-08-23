"""
Is the result a property of the DATA, or of the extractor we wrote?

WHY THIS EXISTS
---------------
Every Phase 1b conclusion - the vanished gain, the sign reversal, the
negative transfer - is produced by
src/metrics/answer_extraction_fallback.py, which this project authored
and then validated with its own manual pass. That is the single largest
attack surface on the paper: a reviewer does not have to show the
extractor is wrong, only that it is load-bearing and self-designed.

The defence is not to argue for our extractor. It is to show the finding
does not depend on it. If several INDEPENDENTLY-MOTIVATED extractors -
including a third-party one and a deliberately crude one with no design
choices at all - all reach the same conclusion, then the conclusion is a
property of the completions, not of our judgement.

THE EXTRACTORS
--------------
1. strict          - "#### N" / \\boxed{N} only. What training actually
                     rewarded, and what produced the original result.
2. verl_flexible   - veRL's own `method="flexible"` from
                     verl/utils/reward_score/gsm8k.py: the LAST number
                     anywhere in the completion, skipping "" and ".".
                     Third-party, written by the RL framework's authors
                     with no knowledge of this experiment.
3. last_line_naive - last number on the last non-empty line. The crudest
                     possible rule; no tie-breaking, no currency
                     handling, no time-stripping, no computed-value
                     preference. Included precisely BECAUSE it is
                     unsophisticated - it has no room for our bias.
4. fallback_chain  - ours (answer_extraction_fallback.py).
5. gt_in_tail      - does the ground-truth value appear as a standalone
                     number in the last 200 characters? Not a real
                     extractor: a deliberately over-permissive UPPER
                     BOUND. It cannot under-count correct answers, so it
                     brackets the others from above.

Reading the output: if `strict` reports a large significant Delta while
2-5 all report ~0, the gain is an artifact of the scoring rule. If the
permissive extractors disagree with EACH OTHER, no conclusion should be
drawn from any of them and the manual review becomes mandatory rather
than confirmatory.
"""

from __future__ import annotations

import re

from src.metrics.answer_extraction import extract_model_answer as _strict
from src.metrics.answer_extraction_fallback import (
    extract_with_fallback as _fallback,
)
from src.metrics.answer_extraction_fallback import normalize_number

# veRL's own pattern, transcribed from verl/utils/reward_score/gsm8k.py.
# Deliberately kept as-is rather than "improved": its value here is that
# somebody else wrote it.
_VERL_NUMBER_RE = re.compile(r"(\-?[0-9\.\,]+)")
_VERL_INVALID = {"", "."}

_PLAIN_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def strict(completion: str) -> str | None:
    return _strict(completion)


def verl_flexible(completion: str) -> str | None:
    """
    veRL's `method="flexible"`: the last number anywhere in the text that
    is not "" or ".". No structural marker required.
    """
    matches = _VERL_NUMBER_RE.findall(completion or "")
    for candidate in reversed(matches):
        if candidate not in _VERL_INVALID:
            return normalize_number(candidate.rstrip("."))
    return None


def last_line_naive(completion: str) -> str | None:
    """
    Last number on the last non-empty line. No heuristics whatsoever -
    this is the version with the least opportunity for author bias.
    """
    lines = [ln for ln in (completion or "").strip().split("\n") if ln.strip()]
    if not lines:
        return None
    matches = _PLAIN_NUMBER_RE.findall(lines[-1])
    return normalize_number(matches[-1]) if matches else None


def fallback_chain(completion: str) -> str | None:
    answer, _method = _fallback(completion or "")
    return answer


def gt_in_tail(completion: str, ground_truth: str, tail_chars: int = 200) -> str | None:
    """
    Over-permissive upper bound: returns the ground truth if it appears as
    a standalone number in the tail. This CANNOT under-count correct
    answers, so it upper-bounds every other extractor. It is not a usable
    scorer - it peeks at the answer - and exists only to bracket the
    others.
    """
    if not ground_truth:
        return None
    text = completion or ""
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    try:
        target = float(str(ground_truth).replace(",", ""))
    except (TypeError, ValueError):
        return None
    for token in _PLAIN_NUMBER_RE.findall(tail):
        try:
            if float(token.replace(",", "")) == target:
                return normalize_number(str(ground_truth))
        except ValueError:
            continue
    return None


# Ordered loosest-last so the printed table reads as a permissiveness ramp.
EXTRACTORS = {
    "strict": lambda c, gt: strict(c),
    "verl_flexible": lambda c, gt: verl_flexible(c),
    "last_line_naive": lambda c, gt: last_line_naive(c),
    "fallback_chain": lambda c, gt: fallback_chain(c),
    "gt_in_tail(upper bound)": gt_in_tail,
}


def score_with_all(df, extractors=None):
    """
    Add one boolean column per extractor to a records DataFrame.
    Returns (df, column_names).
    """
    extractors = extractors or EXTRACTORS
    cols = []
    comp = df["completion"].astype(str).tolist()
    gts = df["ground_truth"].astype(str).tolist()
    norm_gt = [normalize_number(g) for g in gts]

    for name, fn in extractors.items():
        col = f"ok__{name}"
        flags = []
        for c, g, ng in zip(comp, gts, norm_gt):
            # Call the extractor ONCE per row. The earlier version called
            # it twice (once for the None test, once for the comparison),
            # which doubled the cost of a 38,400-row x 5-extractor sweep
            # for no benefit.
            v = fn(c, g)
            flags.append(v is not None and normalize_number(str(v)) == ng)
        df[col] = flags
        cols.append(col)
    return df, cols
