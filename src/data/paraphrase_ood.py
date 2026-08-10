"""
Paraphrase-OOD control data generation (PLAN.md Section 4 item 9 and
Section 10 item 3): a text-only distribution shift (reworded, same
underlying math) used to separate "RL's gain is fragile to modality
specifically" from "RL's gain is fragile to any distribution shift."

STUB IMPLEMENTATION, as specified in the Step 2 build plan: uses a
simple, rule-based surface-form substitution (swap the first proper-noun-
looking name for a different one from a fixed pool) that provably
preserves every number and the required operations, since nothing but a
name token changes. This exercises the control's plumbing end to end but
does not meaningfully vary phrasing beyond that. A more sophisticated
paraphraser is a reasonable future upgrade - tracked as an open item, not
implemented here, so this control's results should be treated as
preliminary until it is upgraded, per PLAN.md Section 10 item 3.

IMPORTANT, same principle as render.py: the model(s) under study (base or
RL-trained) must never be the ones generating these paraphrases. This stub
uses fixed, non-learned string substitution, so there is no risk of that
here at all - but if this is ever upgraded to an LLM-based paraphraser, it
must be a separate model/instance, never pi_base or pi_RL.
"""

from __future__ import annotations

import random
import re

_NAME_POOL = ["Maria", "James", "Aisha", "Wei", "Carlos", "Fatima", "Noah", "Priya"]
_NAME_RE = re.compile(r"\b[A-Z][a-z]+\b")


def paraphrase_stub(question_text: str, seed: int = 0) -> str:
    """
    Deterministic (per question+seed) light rule-based paraphrase: swaps
    every occurrence of the first proper-noun-looking token for a
    different name from a fixed pool. Numbers and sentence structure are
    left completely untouched, so the required math is guaranteed
    unchanged.

    Returns the question unchanged if no name-like token is found (some
    GSM8K problems don't name a person).
    """
    rng = random.Random(f"{question_text}|{seed}")
    match = _NAME_RE.search(question_text)
    if match is None:
        return question_text
    original_name = match.group(0)
    candidates = [n for n in _NAME_POOL if n != original_name]
    replacement = rng.choice(candidates)
    return re.sub(rf"\b{re.escape(original_name)}\b", replacement, question_text)
