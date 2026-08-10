"""
Extracts a model's final numeric answer from its generated text (PLAN.md
Section 4 item 2). Models are prompted to respond using the "#### <number>"
convention (matching GSM8K's own ground-truth format,
src/data/gsm8k_loader.py's extract_final_answer), but a live spot-check
(2026-08-10, scripts/inspect_raw_generations.py, prompted by a
suspiciously low pass@1 from the headroom check) proved that prompting
alone is not reliable: base Qwen2.5-VL-3B-Instruct frequently reasons
correctly but ends with "\\boxed{N}" (a much more common convention in its
pretraining data than GSM8K's specific marker) instead of complying with
the requested format. A "#### only" extractor was silently marking
genuinely correct completions as wrong - not a capability problem, a
measurement problem. This module now tries "####" first (the primary,
requested format), then falls back to "\\boxed{}" before giving up.

Deliberately a separate, lenient function from gsm8k_loader's
extract_final_answer: ground truth should ALWAYS match the "####" pattern
exactly (it comes from the dataset itself, and a mismatch there means
something is badly wrong, so that function raises and has no fallback). A
model's free-form generated text is not guaranteed to comply with any
particular format - a completion matching neither pattern is recorded as
"no answer extracted" (see
src/controls/truncation_check.GenerationRecord.answer_extracted) and
handled by the caller, not treated as an error.

Residual known gap: a completion that states its answer in plain prose
with NEITHER marker (e.g. "Janet makes $18 every day") still won't be
caught - also observed live in the same spot-check. Deliberately not
addressed with a generic "grab the last number in the text" fallback,
since that risks false-positive matches against intermediate calculation
values; PLAN.md's choice of GSM8K specifically over MATH-500 was to avoid
needing that kind of higher-risk, lower-precision extraction. Mitigated
instead by strengthening the prompt to explicitly demand the "####"
format (see the prompt template in the calling scripts) - the fallback
chain here is defense in depth on top of that, not a replacement for it.
"""

from __future__ import annotations

import re

_HASH_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_BOXED_ANSWER_RE = re.compile(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}")


def extract_model_answer(generated_text: str) -> str | None:
    """
    Returns the normalized numeric answer (commas stripped), trying in
    order: the LAST "#### <number>" match, then the LAST "\\boxed{<number>}"
    match. Returns None if neither pattern is found anywhere in the text.

    Takes the LAST match of whichever pattern succeeds (not the first) -
    a model could in principle reference "####" or "\\boxed{}" earlier in
    its reasoning before stating a final answer; the final occurrence is
    the one that reflects what it's actually committing to.
    """
    hash_matches = _HASH_ANSWER_RE.findall(generated_text)
    if hash_matches:
        return hash_matches[-1].replace(",", "").strip()

    boxed_matches = _BOXED_ANSWER_RE.findall(generated_text)
    if boxed_matches:
        return boxed_matches[-1].replace(",", "").strip()

    return None


def is_correct(generated_text: str, ground_truth: str) -> bool:
    """
    True if the model's extracted answer exactly matches the ground
    truth (already-normalized string, e.g. from
    gsm8k_loader.GSM8KExample.final_answer). Returns False (not an
    exception) if no answer could be extracted at all - a missing answer
    is simply incorrect, not a special case.
    """
    extracted = extract_model_answer(generated_text)
    return extracted is not None and extracted == ground_truth
