"""
Truncation-rate check (PLAN.md Phase 1 item 6, Section 5 row 7): measures
how often generations hit the max-token cap (500, per PLAN.md Section 4)
without producing a parseable final answer. This exists specifically to
catch an asymmetric-truncation confound: if the RL model drifts toward
longer outputs (documented "length gaming" reward-hacking behavior, PLAN.md
Section 1.6) more than the base model, it could get cut off more often -
making Delta_text/Delta_pixel partly reflect "who got truncated more,"
not reasoning ability. If truncation is low and symmetric across
base/RL/modality, the 500-token cap is fine as-is; if it's asymmetric,
PLAN.md says to raise the cap and re-run before trusting results.

Two pieces, deliberately decoupled from any specific answer-extraction
implementation (that's Step 7's responsibility) so this module has a
single responsibility - detecting truncation, not parsing answers:
  - hit_max_tokens(): given a generation's raw output length + the
    tokenizer's eos_token_id, determine whether generation was cut off by
    the length cap rather than stopping naturally.
  - compute_truncation_rate(): pure aggregation over caller-supplied
    per-generation records (hit_cap: bool, answer_extracted: bool) -
    Step 6's sampling harness produces these records; this module doesn't
    need to know how "answer_extracted" was determined.
"""

from __future__ import annotations

from dataclasses import dataclass


def hit_max_tokens(output_token_ids, max_new_tokens: int, eos_token_id) -> bool:
    """
    True if a generation was cut off by the length cap rather than
    stopping naturally via EOS. `output_token_ids` is the *generated*
    portion only (prompt tokens already excluded by the caller - Step 6's
    harness, which already slices this out for decoding).

    eos_token_id may be an int or a list/set of ints (some tokenizers have
    multiple valid end tokens); either is accepted.
    """
    if len(output_token_ids) < max_new_tokens:
        # Stopped before hitting the cap at all - must have hit EOS (or
        # some other natural stop condition), not truncation.
        return False

    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    else:
        eos_ids = set(eos_token_id)

    last_token = output_token_ids[-1]
    # Reached the cap AND the final token isn't an EOS token -> truncated.
    # (Reaching the cap exactly on the same step EOS would have appeared
    # is the one ambiguous edge case; treating it as "not truncated" here
    # since the model did produce a natural stop token, just right at the
    # boundary - a generous rather than alarmist reading.)
    return last_token not in eos_ids


@dataclass(frozen=True)
class GenerationRecord:
    """One generation's outcome, as needed for the truncation check."""

    hit_cap: bool
    answer_extracted: bool


def compute_truncation_rate(records: list[GenerationRecord]) -> dict:
    """
    Truncation rate, per PLAN.md's definition: fraction of generations
    that hit the token cap WITHOUT a successfully extracted answer (a
    generation that hits the cap but still happens to contain a parseable
    answer isn't counted as a problem case).
    """
    if not records:
        return {"truncation_rate": 0.0, "n": 0, "truncated_no_answer": 0}

    truncated_no_answer = sum(
        1 for r in records if r.hit_cap and not r.answer_extracted
    )
    return {
        "truncation_rate": truncated_no_answer / len(records),
        "n": len(records),
        "truncated_no_answer": truncated_no_answer,
    }


def check_truncation_symmetry(
    rate_by_cell: dict[str, float], warn_threshold: float = 0.05
) -> dict:
    """
    Compares truncation rates across the base/RL x T/D/E cells (keys are
    caller-chosen labels, e.g. "base_T", "rl_E"). Flags two distinct
    concerns, matching PLAN.md Phase 1 item 6:
      - any single cell's rate exceeding warn_threshold (default 5%)
      - a meaningful asymmetry between the RL and base cells for the same
        condition, which is the confound this check exists to catch
        (caller should pass keys as "{base|rl}_{T|D|E}" for this half of
        the check to run; if that convention isn't followed, only the
        per-cell threshold check applies).
    """
    high_cells = {k: v for k, v in rate_by_cell.items() if v > warn_threshold}

    asymmetries = {}
    for key in rate_by_cell:
        if not key.startswith("rl_"):
            continue
        condition = key[len("rl_") :]
        base_key = f"base_{condition}"
        if base_key in rate_by_cell:
            diff = rate_by_cell[key] - rate_by_cell[base_key]
            if abs(diff) > warn_threshold:
                asymmetries[condition] = {
                    "base": rate_by_cell[base_key],
                    "rl": rate_by_cell[key],
                    "diff": diff,
                }

    return {
        "clean": len(high_cells) == 0 and len(asymmetries) == 0,
        "high_truncation_cells": high_cells,
        "base_vs_rl_asymmetries": asymmetries,
    }
