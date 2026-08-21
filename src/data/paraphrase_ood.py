"""
Paraphrase-OOD control data (PLAN.md Section 4 item 9 and Section 10
item 3): a text-only distribution shift - reworded, same underlying maths
- used to separate "RL's gain is fragile to modality specifically" from
"RL's gain is fragile to any distribution shift."

STATUS (2026-08-22): THE ORIGINAL STUB IS NOT FIT FOR THE REAL CONTROL.
----------------------------------------------------------------------
The first implementation here swapped one proper noun for another from a
fixed pool ("Janet" -> "Maria"), leaving numbers and sentence structure
untouched. Its own docstring said so: it "does not meaningfully vary
phrasing beyond that" and its results "should be treated as preliminary
until it is upgraded."

That upgrade is now done, and it changes the conclusion the control can
support. A name swap is not a distribution shift, so a null result from
it would have meant nothing - and worse, would have looked like evidence.
A reviewer comparing "we tested robustness to rewording" against a diff
showing Janet became Maria would reasonably discount the whole controls
section.

THE REAL IMPLEMENTATION: GSM-PLUS
---------------------------------
`src/data/harder_datasets.py` uses the "problem understanding" subset of
GSM-Plus (Li et al., ACL 2024, arXiv 2402.19255) - genuine rewrites of
GSM8K problems with the answer preserved by construction, produced and
validated by a peer-reviewed source rather than by us.

Verified on the real data (2026-08-22):
  * All 50 of this project's evaluation problems have a rephrased
    variant, so the control runs on exactly the same problems, paired by
    problem_idx against the existing Phase 1 records.
  * All 50 rephrased variants carry the same final answer as their
    original - checked, not assumed.
  * The rewrites are substantial. Example:
      original : "For his 30th birthday, Elvira chose a new computer
                  with many accessories as a gift. She has a budget..."
      rephrased: "Elvira is celebrating her 30th birthday and decides to
                  treat herself to a new computer setup, thanks..."

Use `load_paraphrased_eval_set()` below, or run it end to end with
`scripts/run_variant_eval_on_modal.py --dataset gsmplus_rephrased`.

The stub is retained ONLY because scripts/sanity_check_data.py exercises
it as a plumbing check. It must not be used for a reported result, and
now warns if called.
"""

from __future__ import annotations

import random
import re
import warnings

_NAME_POOL = ["Maria", "James", "Aisha", "Wei", "Carlos", "Fatima", "Noah", "Priya"]
_NAME_RE = re.compile(r"\b[A-Z][a-z]+\b")


def load_paraphrased_eval_set(problems=None, n_problems: int = 50):
    """
    The REAL paraphrase-OOD set: GSM-Plus rewrites of this project's own
    evaluation problems, with problem_idx preserved so the records pair
    directly against the Phase 1 parquets.

    `problems` defaults to the canonical first `n_problems` of the GSM8K
    test split - the same slice Phase 1 evaluated - so the control is
    automatically aligned with the result it is testing rather than
    depending on the caller to keep two slices in sync.

    Returns (examples, stats); `stats` reports coverage and any problem
    that had no variant, so a partial match is a visible number rather
    than a silent shortfall.
    """
    from src.data.gsm8k_loader import load_gsm8k
    from src.data.harder_datasets import load_rephrased_for_problems

    if problems is None:
        problems = load_gsm8k("test")[:n_problems]
    return load_rephrased_for_problems(problems)


def paraphrase_stub(question_text: str, seed: int = 0) -> str:
    """
    DEPRECATED - plumbing check only, never a reported result.

    Swaps every occurrence of the first proper-noun-looking token for a
    different name from a fixed pool. Numbers and sentence structure are
    untouched, so the required maths is guaranteed unchanged - which is
    also precisely why this is too weak to serve as a distribution shift.

    Use load_paraphrased_eval_set() for the actual control.
    """
    warnings.warn(
        "paraphrase_stub() is a name-swap, not a paraphrase, and is not a valid "
        "distribution shift for the paraphrase-OOD control. Use "
        "src.data.paraphrase_ood.load_paraphrased_eval_set() (GSM-Plus) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    rng = random.Random(f"{question_text}|{seed}")
    match = _NAME_RE.search(question_text)
    if match is None:
        return question_text
    original_name = match.group(0)
    candidates = [n for n in _NAME_POOL if n != original_name]
    replacement = rng.choice(candidates)
    return re.sub(rf"\b{re.escape(original_name)}\b", replacement, question_text)
