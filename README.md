# grpo-vlm-modality-shift

Does RLVR's reasoning gain survive a modality shift? Testing the sharpening hypothesis on VLM math reasoning by training Qwen2.5-VL-3B-Instruct with LoRA + Dr. GRPO on **text-only** GSM8K problems, then measuring whether the accuracy gain holds when the same problems are shown as rendered images (Text / Decomposed / End-to-end conditions), via pass@1/pass@k diagnostics with bootstrap CIs across multiple seeds.

Full research design, hypothesis, and validity checklist: see [PLAN.md](PLAN.md) — that document is the single source of truth for *what* and *why*; this README covers *how to run things*.

**Status**: Phase 0 (scaffolding). Nothing has been trained yet.

## Repo layout

```
grpo-vlm-modality-shift/
├── PLAN.md              # research design — source of truth
├── src/
│   ├── data/             # GSM8K-V + GSM8K text loading, leakage check, paraphrase-OOD generation
│   ├── training/          # LoRA + Dr. GRPO training script
│   ├── inference/          # vLLM batched sampling harness, T/D/E condition runners
│   ├── metrics/             # unbiased pass@k estimator, bootstrap CI, per-seed aggregation
│   └── controls/             # manipulation check, difficulty-matching, transcription fidelity, truncation-rate check
├── configs/                # Modal app config, LoRA config, Dr. GRPO config (per-seed variants)
├── scripts/                 # CLI entrypoints
├── notebooks/                # exploratory only, nothing load-bearing
└── tests/
```

## Compute

Modal, GPU per PLAN.md §3/§9:
- A10 — 3B pilot / Phase 0 smoke test
- A100 40GB — required for Phase 1 (3B full run) and any 7B work (Phase 2A instability on 24GB-class GPUs per PLAN.md §3)

## Reproducing each step

Filled in as each build step lands. Nothing to run yet — see PLAN.md §3 for the phase breakdown and exit criteria.
