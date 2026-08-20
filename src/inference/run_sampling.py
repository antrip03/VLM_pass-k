"""
Production-scale orchestration for Step 6: runs src/inference/sampler.py
across many problems x conditions x n samples, for one model checkpoint,
and saves results durably (Parquet) for Step 7's pass@k computation -
the "real" version of what scripts/smoke_test_sampler.py does at toy
scale.

Deliberately model-loading-agnostic: takes an already-loaded (model,
processor) pair rather than a model path, since "how to load the model"
differs by checkpoint type (plain base model vs. the corrected
PEFT-wrapped trained-checkpoint loading built in
scripts/verify_checkpoint_correct_loading.py / scripts/
run_verl_dry_run_on_modal.py) - that decision belongs in the calling
Modal script, not duplicated here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.data.gsm8k_loader import GSM8KExample
from src.inference.sampler import CONDITION_SAMPLERS


@dataclass
class SamplingRecord:
    problem_idx: int
    question: str
    ground_truth: str
    condition: str
    sample_idx: int
    completion: str
    extracted_answer: str | None
    transcription: str | None
    correct: bool


def run_sampling(
    model,
    processor,
    problems: list[GSM8KExample],
    conditions: list[str],
    n: int,
    progress_label: str = "",
    **sampler_kwargs,
) -> list[SamplingRecord]:
    """
    Runs every requested condition for every problem, n samples each.
    conditions: subset of ["T", "D", "E"] (src.inference.sampler.CONDITION_SAMPLERS).
    Returns a flat list of records - one per (problem, condition, sample).
    """
    for c in conditions:
        if c not in CONDITION_SAMPLERS:
            raise ValueError(f"Unknown condition {c!r}, must be one of {list(CONDITION_SAMPLERS)}")

    # Filter kwargs per condition rather than forwarding blindly: the
    # conditions have genuinely different knobs (image_chunk is meaningful
    # for D/E, meaningless for text-only T; transcribe_max_tokens only
    # exists on D), so a single **sampler_kwargs passed to all three
    # raises TypeError on the first condition that doesn't take one. The
    # caller should be able to pass the union of options and have each
    # condition take what applies to it.
    import inspect
    import time

    # Per-problem progress with a live ETA. Evaluation is a multi-hour job
    # whose only external signal was previously "the container is still
    # alive" - there was no way to distinguish steady progress from a hang,
    # the same blind spot that made the training runs hard to reason about
    # until streamed [progress] lines were added there.
    started = time.time()
    total = len(problems)

    def _hms(seconds: float) -> str:
        seconds = int(max(seconds, 0))
        return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"

    records: list[SamplingRecord] = []
    for problem_no, problem in enumerate(problems, start=1):
        for condition in conditions:
            sampler_fn = CONDITION_SAMPLERS[condition]
            accepted = inspect.signature(sampler_fn).parameters
            kwargs = {k: v for k, v in sampler_kwargs.items() if k in accepted}
            samples = sampler_fn(model, processor, problem.question, n=n, **kwargs)
            for sample_idx, sample in enumerate(samples):
                records.append(
                    SamplingRecord(
                        problem_idx=problem.idx,
                        question=problem.question,
                        ground_truth=problem.final_answer,
                        condition=condition,
                        sample_idx=sample_idx,
                        completion=sample.completion,
                        extracted_answer=sample.extracted_answer,
                        transcription=sample.transcription,
                        correct=(sample.extracted_answer == problem.final_answer),
                    )
                )

        elapsed = time.time() - started
        per_problem = elapsed / problem_no
        remaining = (total - problem_no) * per_problem
        correct_so_far = sum(1 for r in records if r.correct)
        tag = f"{progress_label} " if progress_label else ""
        print(
            f"[eval] {tag}problem {problem_no}/{total}"
            f" | elapsed {_hms(elapsed)}"
            f" | {per_problem:.0f}s/problem"
            f" | ETA {_hms(remaining)}"
            f" | running acc {correct_so_far / max(len(records), 1):.3f}",
            flush=True,
        )
    return records


def save_sampling_records(records: list[SamplingRecord], out_path: str) -> None:
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_parquet(out_path)


def load_sampling_records(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
