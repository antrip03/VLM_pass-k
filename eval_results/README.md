# Phase 1 evaluation results (measured 2026-08-20)

Raw outputs of the Phase 1 evaluation described in `PLAN.md` §4. Committed
here rather than left on the Modal Volume they were produced on, because
Volumes are workspace-scoped and this project switched Modal accounts
several times during the run — the same reason checkpoints are mirrored to
the HF Hub (`src/training/hf_checkpoint_sync.py`).

## What was run

- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct`, base vs. Dr. GRPO RL checkpoint
  `global_step_467` (one clean epoch of text-only GSM8K training).
- **Grid**: 50 GSM8K *test* problems × n=128 samples × 3 conditions
  (T/D/E) × 2 models = **38,400 generations**.
- **Cost**: ~3.3 hr per model on one A10G, ~$9 total.

## Files

| File | Contents |
|---|---|
| `analysis.json` | pass@1 and pass@64 per model/condition with bootstrap CIs, plus the headline Δ_text / Δ_pixel / Δ_decomposed and their paired-bootstrap CIs. **The headline result.** |
| `summary_base.json`, `summary_rl.json` | Per-model run summary: record count, wall-clock, per-condition accuracy, per-condition no-answer rate, and the vision-tower SHA-256. |
| `records_base.parquet`, `records_rl.parquet` | One row per generation (19,200 each): problem, ground truth, condition, sample index, **full completion text**, extracted answer, transcription (condition D only), and correctness. Scored with the **corrected** extractor. |
| `records_*.prefix_extractor.parquet` | The same rows scored with the **pre-fix** extractor, kept as an audit trail for the correction described below. |

Because full completion text is stored, extraction and scoring can be
redone offline at zero GPU cost — that is exactly how the correction below
was applied to an already-finished 38,400-generation run
(`scripts/rescore_records.py`).

## The correction these files document

The original extractor required a digit immediately after `####`, so real,
correct completions ending `#### <18>`, `#### \$18`, or `#### $18` were
scored as *no answer*. Rates were 21–27%, and they were **not** truncation:
98.1% of unparsed completions had a digit in their last 120 characters, and
their mean length (806 chars) matched successfully-parsed ones (747).

The rate was **asymmetric between models** — base 21.2% vs RL 5.9% on
condition T — because `src/training/reward_fn.py` shares this extractor, so
RL was directly rewarded for emitting parseable answers. Uncorrected,
Δ_text reads **+0.099**; corrected, **+0.059**. Roughly 40% of the apparent
reasoning gain was format compliance.

Diff the `.prefix_extractor.parquet` files against their counterparts to
see this directly.

## Vision-tower manipulation check

`vision_sha256` is identical in both summaries
(`48ed880e74c758bacf38d0fb424026e0de0aefdd38339156664e8f64b57163cf`),
confirming text-only training left perception byte-identical — so any
measured Δ_pixel comes from changed reasoning, not changed perception
(`PLAN.md` §3 Phase 1 item 8).

## Known residual

Condition E still parses worse than T after the fix (base: 21.0% vs 13.4%
no-answer), so part of the T→E level gap remains measurement rather than
capability. Not yet diagnosed.
