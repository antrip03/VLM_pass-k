"""
Shared prompt templates (PLAN.md Section 1.7's T/D/E conditions all need
a consistent instruction format). Centralized here rather than duplicated
per-script, after finding the hard way that an under-specified format
instruction has real consequences: a live spot-check (2026-08-10,
scripts/inspect_raw_generations.py) found the model frequently reasoning
correctly but ending with "\\boxed{N}" instead of the requested "#### N",
because the original instruction ("give your final answer in the format
'#### <number>'") was too soft to reliably override the model's strong
pretrained habit of using \\boxed{}. src/metrics/answer_extraction.py now
has a \\boxed{} fallback as defense in depth, but the prompt itself should
still ask as unambiguously as possible - the two fixes are complementary,
not redundant.
"""

from __future__ import annotations

TEXT_CONDITION_TEMPLATE = (
    "{question}\n\n"
    "Solve this step by step. Then, on its own final line, write your "
    "answer in EXACTLY this format, with nothing else on that line and "
    "nothing after it:\n"
    "#### <number>"
)


def build_text_prompt(question: str) -> str:
    """Condition T (PLAN.md Section 1.7): the problem as plain text."""
    return TEXT_CONDITION_TEMPLATE.format(question=question)


# Real, live-validated wording (scripts/expanded_length_check_tde.py,
# 2026-08-10 - 280 real generations across T/D/E, no anomalies), moved
# here from that diagnostic script so Step 6's real sampling harness and
# any future diagnostic share one definition instead of duplicating it.
E_CONDITION_INSTRUCTION = (
    "Solve the math problem shown in the image step by step. Then, on its "
    "own final line, write your answer in EXACTLY this format, with "
    "nothing else on that line and nothing after it:\n"
    "#### <number>"
)

D_TRANSCRIBE_INSTRUCTION = (
    "Transcribe the math problem shown in the image exactly as written, "
    "word for word. Do not solve it or explain it - only output the "
    "transcribed problem text."
)


def build_image_prompt() -> str:
    """
    Condition E (PLAN.md Section 1.7, End-to-end): the instruction text
    accompanying the rendered problem image - perceive and reason in one
    continuous pass. No {question} formatting needed (the question is the
    image itself, passed separately as the vision input).
    """
    return E_CONDITION_INSTRUCTION


def build_transcribe_prompt() -> str:
    """Condition D (PLAN.md Section 1.7, Decomposed), step 1: transcribe-only."""
    return D_TRANSCRIBE_INSTRUCTION


def build_decomposed_solve_prompt(transcribed_text: str) -> str:
    """
    Condition D, step 2: feed the model's own transcription back as plain
    text and ask it to solve - deliberately reuses build_text_prompt
    (same instruction, same answer-format requirement) rather than a
    separate template, since at this point it's structurally identical to
    Condition T except the text originated from the model's own
    transcription instead of the ground-truth question.
    """
    return build_text_prompt(transcribed_text)
