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
    uses: no thousands separators, an integral decimal rendered as an
    integer ("75.00" -> "75"), and trailing zeros dropped from a genuine
    decimal ("54.40" -> "54.4").

    The trailing-zero case was found by the manual validation pass
    (2026-08-22): "54.40" and "54.4" are the same number, but comparing
    them as strings scored the extractor wrong. That was a bug in the
    COMPARISON, not in the extraction, and it inflated the measured error
    rate.
    """
    s = token.replace(",", "").strip()
    if "." in s:
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return repr(f).rstrip("0").rstrip(".")
        except ValueError:
            return s
    return s


# Numbers that are part of a clock time ("1:00 PM", "5:00") must not be
# read as answers. Found in validation: three completions ended
# "...burning from 1:00 PM to 5:00 PM", where the last number on the line
# is "00".
_TIME_RE = re.compile(r"\d{1,2}:\d{2}")

# --- fixes from the independent human review (2026-08-23) --------------
# A second reviewer checked 20 traces and found three real defects. All
# three are handled below. On the 9 unambiguous traces the extractor
# already agreed with them 9/9; every disagreement was one of these.

# (1) LaTeX FRACTIONS. The model answers \frac{44}{3} and the last-number
#     scan returned the DENOMINATOR, "3" - worse than returning nothing,
#     because a ground truth of 3 would score it correct for the wrong
#     reason. Fractions in these completions are always properly formed
#     LaTeX, so they can be evaluated exactly. This also RESCUES answers:
#     \frac{36}{2} against a ground truth of 18 is correct and was
#     previously missed.
_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}\s*\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}")

# (2) FENCED CODE BLOCKS. A completion ending in a python block was read
#     as an answer by scanning the code itself ("total_cost = (4*5) +
#     (4*3) + (4*3)" -> "3"). Code that was never executed states no
#     answer. Stripped before the line scan; the code_block rule above
#     still catches a fence whose entire content is one number.
_FENCE_BLOCK_RE = re.compile(r"```.*?```", re.S)

# (3) UNEVALUATED EXPRESSIONS AND STEP HEADERS. Two shapes that contain
#     digits but assert no answer:
#       "#### Corrected Result: 13 - (Number of Legos Sold ...)"  -> "13"
#       "Step 5: Simplify the calculation to find ..."            -> "5"
#     The first is a number followed by an operator and then a word or
#     bracket - an expression the model never resolved. The second is a
#     numbered step heading.
_UNRESOLVED_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*[-+*/×÷]\s*[\(\[A-Za-z]")
_STEP_HEADER_RE = re.compile(r"^\s*\**\s*step\s+\d+\s*[:.\)]", re.I)

# Numbers that appear as the RESULT of a computation ("= 435", "= 8 cm").
_EQUALS_RESULT_RE = re.compile(rf"=\s*\\?\$?\s*({_NUMBER})")


def _eval_fraction(numerator: str, denominator: str) -> str | None:
    """
    Evaluate a LaTeX fraction to this project's canonical numeric string.

    An integral result is returned as an integer (\\frac{36}{2} -> "36/2"
    -> "18"), so a correct fractional answer now scores correct. A
    non-integral result is returned as a decimal, which simply will not
    match GSM8K's integer ground truths - the correct outcome, and far
    better than the previous behaviour of returning the denominator.
    """
    try:
        num = float(numerator.replace(",", ""))
        den = float(denominator.replace(",", ""))
    except ValueError:
        return None
    if den == 0:
        return None
    return normalize_number(repr(num / den))


def _numbers_outside_times(text: str) -> list[str]:
    """All numbers in `text`, with clock-time components removed first."""
    return _NUMBER_RE.findall(_TIME_RE.sub(" ", text))


def _last_number_in(text: str, computed: set[str] | None = None) -> str | None:
    """
    Best final-answer number on one line.

    Plain "last number on the line" fails on a TRAILING QUALIFIER, which
    the manual validation showed is the dominant error mode:

        "John is 435 miles from home at the end of those 4 hours."  -> 4
        "Claire will eat 7 dozens of eggs in 4 weeks."              -> 4
        "Mike scored a total of 9 points over the 40-minute period" -> 40

    In every such case the true answer had appeared just above as the
    result of a computation ("= 435 miles", "= 7", "= 9 points"), whereas
    the qualifier had not. So when `computed` (numbers following an "=" in
    the surrounding tail) is supplied, prefer the last number on the line
    that is also a computed result, and fall back to positional order
    only when none matches.
    """
    candidates = [normalize_number(t) for t in _numbers_outside_times(text)]
    if not candidates:
        return None
    if computed:
        matching = [c for c in candidates if c in computed]
        if matching:
            return matching[-1]
    return candidates[-1]


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

    # The code_block rule needs the ORIGINAL text, since it looks for a
    # fence whose entire content is a number.
    raw_lines = [ln for ln in generated_text.strip().split("\n") if ln.strip()]
    if not raw_lines:
        return None, "empty"
    raw_tail = "\n".join(raw_lines[-max_lines_back:])

    m = _CODE_BLOCK_RE.search(raw_tail)
    if m:
        return normalize_number(m.group(1)), "code_block"

    # Everything after this point scans PROSE, so fenced code is removed
    # first: unexecuted code states no answer, and scanning it invents one
    # ("total_cost = (4*5) + (4*3)" was read as "3").
    text_no_code = _FENCE_BLOCK_RE.sub(" ", generated_text)
    lines = [ln for ln in text_no_code.strip().split("\n") if ln.strip()]
    if not lines:
        return None, "none"

    tail_lines = lines[-max_lines_back:]
    tail = "\n".join(tail_lines)

    # LaTeX fraction anywhere in the tail. Checked before the positional
    # rules because a fraction is an explicit answer form, whereas "last
    # number on the line" is a fallback guess.
    frac = _FRAC_RE.findall(tail)
    if frac:
        value = _eval_fraction(*frac[-1])
        if value is not None:
            return value, "latex_fraction"

    for pattern in _ANSWER_MARKERS:
        found = pattern.findall(tail)
        if found:
            return normalize_number(found[-1]), "answer_marker"

    m = _BOLD_RE.findall(tail)
    if m:
        return normalize_number(m[-1]), "bold"

    # Numbers that appeared as computation results anywhere in the tail,
    # used to disambiguate a concluding sentence that also carries a
    # trailing qualifier (see _last_number_in).
    computed = {normalize_number(t) for t in _EQUALS_RESULT_RE.findall(tail)}

    # Last content-bearing line, scanning backwards past pure
    # LaTeX/markdown noise such as a lone "\]". Within a line, a
    # currency-marked number wins over mere position (see _CURRENCY_RE).
    for line in reversed(tail_lines):
        if _NOISE_RE.match(line):
            continue
        # A numbered step heading is never a conclusion.
        if _STEP_HEADER_RE.match(line):
            continue
        # A number followed by an operator and then a word/bracket is an
        # expression the model never resolved, not an answer.
        if _UNRESOLVED_RE.search(line):
            continue
        money = _CURRENCY_RE.findall(_TIME_RE.sub(" ", line))
        if money:
            return normalize_number(money[-1]), "currency"
        value = _last_number_in(line, computed)
        if value is not None:
            return value, "last_line"

    return None, "none"


def is_correct_with_fallback(generated_text: str, ground_truth: str) -> tuple[bool, str]:
    """(correct, method) using the fallback chain."""
    answer, method = extract_with_fallback(generated_text)
    if answer is None:
        return False, method
    return normalize_number(str(ground_truth)) == answer, method
