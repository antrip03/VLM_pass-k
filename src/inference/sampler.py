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
    model,
    processor,
    prompt_text: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    text_chunk: int | None = None,
) -> list[str]:
    """
    Text-only generation, batched via num_return_sequences - validated
    safe at real concurrency (scripts/vram_smoke_test_2000cap.py: 64
    concurrent sequences, 10.98 GB peak on a 40GB A100, huge margin).

    Chunked with OOM fallback (2026-08-20): that 40GB margin does NOT
    carry to smaller cards. At the eval's real n=128, KV cache alone is
    ~8.5GB on top of ~7.5GB of weights, which is comfortable on an A100
    but close to the edge of a 24GB A10G. Rather than assume, generate in
    chunks and halve on CUDA OOM, mirroring _generate_image_batch. Default
    (None) keeps the original single-call behaviour so A100 runs are
    unchanged.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    pad_id = processor.tokenizer.pad_token_id

    completions: list[str] = []
    remaining = n
    chunk = n if text_chunk is None else max(1, min(text_chunk, n))
    while remaining > 0:
        this_chunk = min(chunk, remaining)
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=this_chunk,
                )
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            torch.cuda.empty_cache()
            print(f"[sampler] text generation OOM - halving chunk to {chunk} and retrying")
            continue
        for seq in out:
            generated = seq[prompt_len:]
            nonpad = generated[generated != pad_id]
            completions.append(processor.tokenizer.decode(nonpad, skip_special_tokens=True))
        remaining -= this_chunk
    return completions


DEFAULT_IMAGE_CHUNK = 16


def _generate_image_batch(
    model,
    processor,
    image,
    instruction: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    image_chunk: int = DEFAULT_IMAGE_CHUNK,
) -> list[str]:
    """
    Image-conditioned generation, in CHUNKS of `image_chunk` samples per
    generate() call (2026-08-20, was one-sample-per-call).

    The original one-at-a-time loop was a correct response to a real OOM
    (scripts/expanded_length_check_tde.py, 2026-08-10: num_return_sequences
    OOMs the vision encoder's attention, because HF's generate() replicates
    pixel_values BEFORE the vision forward pass) - but "n=1" and "n=128"
    are not the only options, and the cost of the safest one is severe:
    at n=128 it means 128 sequential decodes per problem per condition,
    which is what makes single-sequence throughput (~tens of tok/s) rather
    than batched throughput (~865 tok/s aggregate, PLAN.md 9.2) the
    binding constraint on evaluation cost. A modest chunk keeps the vision
    tower's replicated activations bounded while recovering most of the
    batching win; chunk size is a parameter precisely because the safe
    ceiling is empirical, not derivable - calibrate it, don't assume it.

    Falls back automatically: on a CUDA OOM the chunk is halved and
    retried, down to 1 (the original, known-safe behaviour), so a
    too-large chunk degrades performance instead of losing the run.
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

    completions: list[str] = []
    remaining = n
    chunk = max(1, min(image_chunk, n))
    while remaining > 0:
        this_chunk = min(chunk, remaining)
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=this_chunk,
                )
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            torch.cuda.empty_cache()
            print(f"[sampler] image generation OOM - halving chunk to {chunk} and retrying")
            continue
        for seq in out:
            generated = seq[prompt_len:]
            nonpad = generated[generated != pad_id]
            completions.append(processor.tokenizer.decode(nonpad, skip_special_tokens=True))
        remaining -= this_chunk
    return completions


def _generate_text_multi_prompt(
    model,
    processor,
    prompt_texts: list[str],
    max_new_tokens: int,
    temperature: float,
    batch_size: int = 16,
) -> list[str]:
    """
    One completion for each of many DIFFERENT prompts, batched with left
    padding (2026-08-20). Condition D needs exactly this: each of its n
    samples has its own distinct transcription, so num_return_sequences
    (same prompt, n outputs) does not apply, and the previous code fell
    back to n separate single-sample calls - the slowest possible path.

    Left padding is required for decoder-only generation: with right
    padding the pads sit between the prompt and the first generated token
    and corrupt the continuation. The tokenizer's padding_side is set
    explicitly here rather than trusted, then restored.
    """
    tokenizer = processor.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    completions: list[str] = []
    try:
        for start in range(0, len(prompt_texts), batch_size):
            batch = prompt_texts[start : start + batch_size]
            prompts = [
                processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": t}]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in batch
            ]
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature
                )
            for seq in out:
                generated = seq[prompt_len:]
                nonpad = generated[generated != pad_id]
                completions.append(tokenizer.decode(nonpad, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = original_side
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
    image_chunk: int = DEFAULT_IMAGE_CHUNK,
    instruction_override: str | None = None,
) -> list[SampleResult]:
    """
    Condition E: n samples, image shown, perceive+reason in one pass.

    instruction_override: for the compliance-matching causal control
    (2026-08-25) - pass build_image_prompt_fewshot() to test whether
    raising format compliance alone, with no training and no content
    change, collapses the strict Delta. None (default) uses the standard
    instruction, unchanged from every prior Phase 1/1b run.
    """
    image = render_problem_image(question)
    instruction = instruction_override or build_image_prompt()
    completions = _generate_image_batch(
        model, processor, image, instruction, n, max_new_tokens, temperature, image_chunk=image_chunk
    )
    return [SampleResult(c, extract_model_answer(c)) for c in completions]


def sample_condition_d(
    model,
    processor,
    question: str,
    n: int,
    transcribe_max_tokens: int = DEFAULT_TRANSCRIBE_MAX_TOKENS,
    solve_max_tokens: int = DEFAULT_SOLVE_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    image_chunk: int = DEFAULT_IMAGE_CHUNK,
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
        model, processor, image, transcribe_instruction, n, transcribe_max_tokens, temperature,
        image_chunk=image_chunk,
    )

    solve_prompts = [build_decomposed_solve_prompt(t) for t in transcriptions]
    solve_completions = _generate_text_multi_prompt(
        model, processor, solve_prompts, solve_max_tokens, temperature
    )
    return [
        SampleResult(completion, extract_model_answer(completion), transcription=transcription)
        for completion, transcription in zip(solve_completions, transcriptions)
    ]


CONDITION_SAMPLERS = {
    "T": sample_condition_t,
    "D": sample_condition_d,
    "E": sample_condition_e,
}
