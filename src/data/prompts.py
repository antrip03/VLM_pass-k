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
