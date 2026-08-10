"""
Sampling harness for PLAN.md Section 1.7's three conditions (Step 6):
  T (Text): problem as typed text.
  D (Decomposed): image -> transcribe-only -> feed transcription back as
    text -> solve.
  E (End-to-end): image -> perceive and reason in one continuous pass.

Reuses generation patterns already validated with real Modal runs this
session, not reinvented:
  - Text-only generation batches safely via num_return_sequences
    (confirmed working up to 64+ concurrent sequences,
    scripts/vram_smoke_test_2000cap.py).
  - Image-conditioned generation does NOT use num_return_sequences - a
    live run (scripts/expanded_length_check_tde.py, 2026-08-10) found
    that OOMs the vision encoder's attention, because HF's generate()
    replicates pixel_values BEFORE the vision forward pass, not after.
    Loops one sample per call instead - slower, but safe regardless of n.
  - The real, live-tested prompt templates from src/data/prompts.py
    (E_CONDITION_INSTRUCTION / D_TRANSCRIBE_INSTRUCTION - the exact
    wording already exercised across 280 real generations with no
    anomalies), not new/untested wording.

Answer extraction reuses src/metrics/answer_extraction.py's
extract_model_answer() (the #### + \\boxed{} fallback chain) - the same
function used for training reward (src/training/reward_fn.py), keeping
training and evaluation scoring consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.prompts import (
    build_decomposed_solve_prompt,
    build_image_prompt,
    build_text_prompt,
    build_transcribe_prompt,
)
from src.data.render import render_problem_image
from src.metrics.answer_extraction import extract_model_answer

DEFAULT_TEMPERATURE = 0.8
DEFAULT_SOLVE_MAX_TOKENS = 1200  # PLAN.md Section 3 Phase 1 item 6, real-data-derived cap
DEFAULT_TRANSCRIBE_MAX_TOKENS = 300  # transcription is short; generous margin (real data: <110 tokens)


@dataclass
class SampleResult:
    completion: str
    extracted_answer: str | None
    transcription: str | None = None  # only populated for Condition D


def _generate_text_batch(
    model, processor, prompt_text: str, n: int, max_new_tokens: int, temperature: float
) -> list[str]:
    """
    Text-only generation, batched via num_return_sequences - validated
    safe at real concurrency (scripts/vram_smoke_test_2000cap.py: 64
    concurrent sequences, 10.98 GB peak on a 40GB A100, huge margin).
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    pad_id = processor.tokenizer.pad_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, num_return_sequences=n
        )
    completions = []
    for seq in out:
        generated = seq[prompt_len:]
        nonpad = generated[generated != pad_id]
        completions.append(processor.tokenizer.decode(nonpad, skip_special_tokens=True))
    return completions


def _generate_image_batch(model, processor, image, instruction: str, n: int, max_new_tokens: int, temperature: float) -> list[str]:
    """
    Image-conditioned generation. Deliberately one sample per generate()
    call, NOT num_return_sequences - see module docstring for why.
    """
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[prompt], images=vision_inputs, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    pad_id = processor.tokenizer.pad_token_id

    completions = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        generated = out[0, prompt_len:]
        nonpad = generated[generated != pad_id]
        completions.append(processor.tokenizer.decode(nonpad, skip_special_tokens=True))
    return completions


def sample_condition_t(
    model,
    processor,
    question: str,
    n: int,
    max_new_tokens: int = DEFAULT_SOLVE_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[SampleResult]:
    """Condition T: n samples of the problem as plain text."""
    prompt = build_text_prompt(question)
    completions = _generate_text_batch(model, processor, prompt, n, max_new_tokens, temperature)
    return [SampleResult(c, extract_model_answer(c)) for c in completions]


def sample_condition_e(
    model,
    processor,
    question: str,
    n: int,
    max_new_tokens: int = DEFAULT_SOLVE_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[SampleResult]:
    """Condition E: n samples, image shown, perceive+reason in one pass."""
    image = render_problem_image(question)
    instruction = build_image_prompt()
    completions = _generate_image_batch(model, processor, image, instruction, n, max_new_tokens, temperature)
    return [SampleResult(c, extract_model_answer(c)) for c in completions]


def sample_condition_d(
    model,
    processor,
    question: str,
    n: int,
    transcribe_max_tokens: int = DEFAULT_TRANSCRIBE_MAX_TOKENS,
    solve_max_tokens: int = DEFAULT_SOLVE_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[SampleResult]:
    """
    Condition D: n independent (transcribe, then solve) pairs. Each
    sample gets its OWN real transcription (not one shared transcription
    reused n times) - this is what actually lets Condition D isolate
    transcription-error variance from reasoning-error variance across
    the sample pool, matching PLAN.md Section 1.7's stated purpose.
    """
    image = render_problem_image(question)
    transcribe_instruction = build_transcribe_prompt()
    transcriptions = _generate_image_batch(
        model, processor, image, transcribe_instruction, n, transcribe_max_tokens, temperature
    )

    results = []
    for transcription in transcriptions:
        solve_prompt = build_decomposed_solve_prompt(transcription)
        solve_completions = _generate_text_batch(model, processor, solve_prompt, 1, solve_max_tokens, temperature)
        completion = solve_completions[0]
        results.append(SampleResult(completion, extract_model_answer(completion), transcription=transcription))
    return results


CONDITION_SAMPLERS = {
    "T": sample_condition_t,
    "D": sample_condition_d,
    "E": sample_condition_e,
}
