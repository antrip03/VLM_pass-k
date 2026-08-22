"""
Fallback answer extraction for EVALUATION ONLY - reading a final answer
out of a completion that never emitted the requested "#### N" marker.

WHY THIS IS NEEDED (measured, not assumed)
-------------------------------------------
Commit 43279ab found that the extractor rejected decorated markers
("#### <18>", "#### \\$18") and fixed that, moving Delta_text from +0.099
to +0.059. That fix was real but addressed only a small slice of the
problem. Measured on the committed Phase 1 records (2026-08-22):

    condition T, base: 856 unparsed (13.4%), of which
        38  (4.4%)  contain "####" somewhere  <- the decoration class
        818 (95.6%) contain NO marker at all  <- NOT addressed by 43279ab

    condition E, base: 1341 unparsed (21.0%), 96.5% with no marker.

So ~95% of unparsed completions are not an extractor bug at all: the
model simply did not follow the output-format instruction, and ended with
a natural-language conclusion instead.

That matters because the failure rate is strongly ASYMMETRIC between
models - base 13.4% vs RL 4.1% on T, base 21.0% vs RL 6.5% on E -
because src/training/reward_fn.py rewarded parseable answers, so RL was
trained into compliance. Inspecting real examples confirms many of those
base completions reason correctly and simply write the answer plainly:

    "Therefore, Janet makes **$18** every day at the farmers' market."
    "So, the boots cost **$104**."
    "Thus, the distance Henry traveled ... is 25 miles."

Scoring those as wrong measures FORMATTING, not reasoning - and the
paper's claim is about reasoning.

WHY THIS IS A SEPARATE MODULE
-----------------------------
src/metrics/answer_extraction.py is shared with src/training/reward_fn.py.
Loosening it there would retroactively change what the reward meant.
This module is imported only by evaluation/rescoring code, so the strict
definition that training actually used stays intact and the two numbers
remain comparable.

THE POINT IS TO REPORT BOTH, NOT TO PICK THE BIGGER ONE
--------------------------------------------------------
A permissive extractor can always manufacture a nicer number, which is
exactly the failure mode this project already suffered once. So the
intended use is to report the STRICT and PERMISSIVE deltas side by side.
The difference between them is itself the finding: it quantifies how much
of the measured gain is instruction-following rather than reasoning.

VALIDATION IS MANDATORY BEFORE USE
----------------------------------
This module must NOT be trusted on its own output. Use
scripts/build_extraction_review_pack.py to hand-label a sample and
measure this extractor's precision against those labels before any number
derived from it goes in the paper.
"""

from __future__ import annotations

import re

from src.metrics.answer_extraction import extract_model_answer as extract_strict

# A number, tolerating thousands separators and decimals. Deliberately
# does NOT allow a leading "-" inside prose: GSM8K answers are effectively
# never negative, and a bare hyphen in text ("16 - 11 = 5") would
# otherwise be captured as part of the number.
_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_NUMBER_RE = re.compile(_NUMBER)

# Explicit answer announcements, checked before falling back to
# "last number in the last line".
_ANSWER_MARKERS = [
    re.compile(rf"(?:final\s+)?answer\s*(?:is|:)\s*\**\s*\\?\$?\s*({_NUMBER})", re.I),
    re.compile(rf"\bis\s+(?:exactly\s+)?\**\s*\\?\$?\s*({_NUMBER})\s*\**\s*[.!]?\s*$", re.I),
]

# A fenced code block whose entire content is one number - observed in
# real completions as "```\n18\n```".
_CODE_BLOCK_RE = re.compile(rf"```[a-z]*\s*({_NUMBER})\s*```", re.I)

# Markdown-bold number, with optional currency: **$18**, **230 miles**.
_BOLD_RE = re.compile(rf"\*\*\s*\\?\$?\s*({_NUMBER})")

# A currency-marked number: $75.00, \$18. Checked BEFORE the bare
# "last number on the line" rule because of a real failure case found in
# the committed records:
#     "So, Terry spends exactly $75.00 on yogurt over 30 days."
# The answer is 75, but the LAST number on that line is 30. Where a line
# marks a quantity as money, that marking is a far better signal of which
# number is the answer than position is.
_CURRENCY_RE = re.compile(rf"\\?\$\s*({_NUMBER})")

# LaTeX/markdown noise stripped before the last-line scan, so "\]" or
# "\[" lines do not read as content-bearing.
_NOISE_RE = re.compile(r"^[\\\[\]\(\)\s`*_$]+$")


def normalize_number(token: str) -> str:
    """
    Canonicalise an extracted number to the same form GSM8K ground truth
    uses: no thousands separators, and an integral decimal rendered as an
    integer ("75.00" -> "75") so it compares equal to the reference.
    """
    s = token.replace(",", "").strip()
    if "." in s:
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
        except ValueError:
            return s
    return s


def _last_number_in(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    return normalize_number(matches[-1]) if matches else None


def extract_with_fallback(
    generated_text: str, max_lines_back: int = 4
) -> tuple[str | None, str]:
    """
    Extract a final answer, returning (answer, method).

    `method` names which rule fired, so every rescored record carries an
    audit trail and the permissive numbers can be broken down by how they
    were obtained rather than being a single opaque total.

    Ordered chain, strictest first:
      1. "strict"      - the shared #### / \\boxed extractor (unchanged)
      2. "code_block"  - a fenced block containing only a number
      3. "answer_marker" - "Answer: N", "the answer is N", "... is N."
      4. "bold"        - a markdown-bold number in the final lines
      5. "last_line"   - last number on the last content-bearing line
      6. None          - genuinely no answer found

    Rules 2-5 are applied only to the TAIL of the completion (the last
    `max_lines_back` content-bearing lines). Scanning the whole text would
    routinely capture intermediate working, which is the failure mode that
    makes naive "last number anywhere" extraction untrustworthy.
    """
    if not generated_text:
        return None, "empty"

    strict = extract_strict(generated_text)
    if strict is not None:
        return strict, "strict"

    lines = [ln for ln in generated_text.strip().split("\n") if ln.strip()]
    if not lines:
        return None, "empty"

    tail_lines = lines[-max_lines_back:]
    tail = "\n".join(tail_lines)

    m = _CODE_BLOCK_RE.search(tail)
    if m:
        return normalize_number(m.group(1)), "code_block"

    for pattern in _ANSWER_MARKERS:
        found = pattern.findall(tail)
        if found:
            return normalize_number(found[-1]), "answer_marker"

    m = _BOLD_RE.findall(tail)
    if m:
        return normalize_number(m[-1]), "bold"

    # Last content-bearing line, scanning backwards past pure
    # LaTeX/markdown noise such as a lone "\]". Within a line, a
    # currency-marked number wins over mere position (see _CURRENCY_RE).
    for line in reversed(tail_lines):
        if _NOISE_RE.match(line):
            continue
        money = _CURRENCY_RE.findall(line)
        if money:
            return normalize_number(money[-1]), "currency"
        value = _last_number_in(line)
        if value is not None:
            return value, "last_line"

    return None, "none"


def is_correct_with_fallback(generated_text: str, ground_truth: str) -> tuple[bool, str]:
    """(correct, method) using the fallback chain."""
    answer, method = extract_with_fallback(generated_text)
    if answer is None:
        return False, method
    return normalize_number(str(ground_truth)) == answer, method
