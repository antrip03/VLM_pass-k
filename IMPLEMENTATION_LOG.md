# Implementation Log: What's Built, What's Validated, and Every Real Bug Along the Way

**Purpose of this document**: `PLAN.md` is the research design (the *why*). This is the engineering
companion (the *how*, and *what actually broke when we tried*). If you're picking this repo up fresh
— human or Claude — read this before touching `configs/modal_app_verl.py` or `src/training/`, because
you will otherwise re-discover several of these bugs the hard way, on real (billed) Modal compute.

Everything described here was validated with **real Modal execution**, not local mocks or assumptions.
Every bug below was found live, diagnosed from the actual error, and fixed with a verified re-run — not
guessed at. Where a fix's correctness wasn't independently confirmed, that's said explicitly.

Branch: `Guneesh`. Status as of this document: PLAN.md's Phase 0 (Steps 1–7, the full scaffolding +
validation pipeline) is complete. The real Phase 1 scientific run has not started yet.

---

## 1. The three Modal environments, and why there are three

| File | Purpose | Status |
|---|---|---|
| `configs/modal_app.py` | Plain HuggingFace/transformers environment. Used for data pipeline work, smoke tests, and any diagnostic that doesn't need veRL/Unsloth's own training loop. | Validated (Step 1) |
| `configs/modal_app_verl.py` | The real training environment: veRL + vLLM + FSDP. This is what Phase 1 will actually run on. | Validated end-to-end (Step 5) |
| `configs/modal_app_unsloth.py` | A comparison probe only, built to answer "would Unsloth be faster?" **Not used for the real pipeline** — see §5. | Validated to run, but not adopted |

They're kept separate deliberately: veRL and Unsloth have mutually incompatible dependency stacks, and
none of Steps 1–4's work needed either of them.

---

## 2. Step-by-step status

| Step | What | Real validation |
|---|---|---|
| 1 | Modal environment | Smoke test passing (GPU detect, model load, text+image generation) |
| 2 | Data pipeline (`src/data/`) | GSM8K loading, self-rendering (pixel-width-correct), leakage checks |
| 3 | Manipulation check + truncation check (`src/controls/`) | Real forward+backward+optimizer step, vision bytes confirmed identical |
| 4 | Headroom pre-check (`src/controls/headroom_check.py`) | Real run: pass@1=0.513, pass@16=0.912 on 25 GSM8K problems, n=64 |
| 5 | LoRA + Dr. GRPO training (`src/training/`) | Real veRL training run, vision provably byte-identical after training, coherent output — see §4 for the full bug log |
| 6 | T/D/E sampling harness (`src/inference/`) | Real generations across all three conditions, accurate transcriptions, clean Parquet persistence |
| 7 | Metrics (`src/metrics/`) | pass@k, bootstrap CI, transcription fidelity — all checked against hand-computed reference values |

Local (no-GPU) sanity checks live in `scripts/sanity_check_*.py` — run these first after any change to
`src/metrics/`, `src/controls/`, or `src/training/reward_fn.py`, they're fast and catch a lot.

---

## 3. Key decisions and their real justification

- **Response length cap: 1200 tokens** (not 500, not 2000). Real data:
  `scripts/check_natural_generation_length.py` (60 generations, text-only) then
  `scripts/expanded_length_check_tde.py` (280 generations across T/D/E) — max observed completion length
  *anywhere, in any condition* was 615 tokens. 1200 leaves ~2x margin. 500 truncated 10% of real
  completions (a real confound risk); 2000 was proven safe but unnecessarily expensive.
- **Seeds: 1** (not 3). User's explicit final decision, not a default. This trades away
  characterizing seed-to-seed variance — recorded as a named limitation in `PLAN.md` §10 item 8, not
  silently dropped.
- **veRL over Unsloth**: see §5.
- **Dr. GRPO, not vanilla GRPO**: `PLAN.md` §7 has the full reasoning. Concretely enabled via two real
  veRL flags, both verified against veRL's actual current source (not docs, which can lag):
  - `actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm` (length-bias fix)
  - `actor_rollout_ref.actor.fsdp_config.use_orig_params=True` — this one is **not** a Dr. GRPO flag
    per se, but is *load-bearing for correctness* (see bug #10 below) and must not be removed.
- **Reward function**: custom (`src/training/reward_fn.py`), not veRL's built-in GSM8K reward. veRL's
  default `method="strict"` only matches `#### N` — verified against real source
  (`verl/utils/reward_score/gsm8k.py`) to have the exact same gap this project already found and fixed
  in Step 4 (models frequently answer via `\boxed{N}` instead).

---

## 4. Step 5's full bug log — read this before re-running training

Getting a real veRL training run working took **eleven** real, independently-diagnosed bugs. Listed in
the order encountered. If you're re-running this on a fresh Modal image or a different veRL commit,
expect some of these to resurface (veRL is a fast-moving, actively-developed framework) — the diagnostic
method that found each one is as valuable as the fix itself.

1. **Missing `MODEL_ID`** in `configs/modal_app_verl.py` — never needed before because the original
   smoke test didn't load a real model. Trivial, caught immediately at import time.
2. **`.sh` script silently not mounted.** `add_local_python_source("src", "configs")` only mounts `.py`
   files — a shell script placed in the same tree never made it into the container. Fix: converted the
   training launch config from a shell script (`run_dry_run.sh`) to a plain Python arg-list builder
   (`src/training/dry_run_config.py::build_args()`), sidestepping the whole class of problem rather than
   finding a different way to ship a non-Python file.
3. **vLLM too old for LoRA rollout serving.** The pinned Docker image (`verlai/verl:vllm011.latest`,
   vllm 0.11.0) raised `"vllm version 0.11.0 not supported... Currently supported vllm versions are
   0.18.0+"` when validating the LoRA+vLLM-rollout code path. Fixed by switching to
   `verlai/verl:vllm024.dev2` (checked real available tags on Docker Hub, not guessed).
4. **GPU memory OOM at vLLM engine startup** — 37 GB of 39.5 GB already used before vLLM even started
   sizing its KV cache. Root cause: `actor_rollout_ref.actor.fsdp_config.param_offload` and
   `optimizer_offload` were both `False` (correct for the *official 8-GPU example*, where actor and
   rollout run on separate GPU groups — wrong for our single, colocated GPU, where the actor's full
   weights+optimizer then permanently compete with vLLM for the same fixed memory pool). Fixed by
   enabling both offload flags.
5. **fp32-by-default model dtype.** veRL's actual `FSDPEngineConfig.model_dtype` defaults to `"fp32"`,
   not `"bf16"` — confirmed via real source, not the (documented-unreliable) veRL docs. Fixed explicitly:
   `actor_rollout_ref.actor.fsdp_config.model_dtype=bf16` (and the `ref.` equivalent).
6. **`qwen2_vl.py`'s "fake ViT" patch assumed the wrong return type.** For text-only training, veRL runs
   a dummy all-zero image through the real vision encoder and multiplies by exactly `0.0` (a standard
   FSDP trick — "touches" the vision parameters so gradient sync doesn't complain about unused
   parameters, with zero actual numerical effect). That code assumed `model.visual(...)` returns a raw
   tensor; the installed transformers version returns a `BaseModelOutputWithPooling` object instead —
   `AttributeError: ... has no attribute 'mean'`. Fixed with a targeted `sed` patch to veRL's source
   inside the Docker image build (`configs/modal_app_verl.py`), unwrapping `.last_hidden_state` if
   present.
7. **Same file, one line earlier: `model.visual` doesn't exist.** The real vision tower lives at
   `model.model.visual` in the installed transformers version (confirmed via direct introspection, not
   assumed) — another real HF class-hierarchy version drift. Fixed with a second, similarly-targeted
   `sed` patch.
8. **vLLM sized its KV cache for the model's full native context (128,000 tokens)** instead of our
   actual ~1,800-token budget — `"4.39 GiB KV cache is needed... larger than the available (4.23 GiB)"`.
   Fixed by explicitly setting `actor_rollout_ref.rollout.max_model_len` to
   `max_prompt_length + max_response_length + margin`.
9. **Near-miss OOM during weight sync** (148 MiB short out of 39.49 GiB) when syncing freshly-trained
   weights from the actor back into vLLM after a real gradient step — a documented peak-memory moment in
   colocated actor+rollout setups. Fixed with two small, targeted adjustments: `gpu_memory_utilization`
   nudged down slightly, and `free_cache_engine=True` (releases vLLM's KV cache when not actively
   generating, freeing room specifically during the training/sync phase).
10. **The big one: FSDP's `use_orig_params` default (`False`) silently overrides `exclude_modules`.**
    After training "succeeded" (`returncode: 0`), independent verification found vision weights had
    changed and completions were pure gibberish — looked exactly like a scientific-validity failure.
    Real mechanism: when `use_orig_params=False`, PyTorch FSDP requires every parameter *flattened into
    the same shard* to share one `requires_grad` value. If a trainable LoRA parameter ends up sharded
    with the (supposedly frozen) vision tower, the *whole shard* gets forced to `requires_grad=True`,
    bypassing `exclude_modules='.*visual.*'` at a level below where LoRA's own config even operates.
    Fixed: `actor_rollout_ref.actor.fsdp_config.use_orig_params=True` (and `ref.` equivalent) — the
    officially-documented PyTorch fix for exactly this LoRA+FSDP combination. **This did not fully solve
    the symptom** — see #11.
11. **The real root cause of #10's symptom was actually a checkpoint-*loading* bug, not training.**
    After applying fix #10, vision still showed as "changed." Direct inspection of the **raw** FSDP
    checkpoint's tensors (not the exported one) found the LoRA deltas were completely healthy — small,
    non-NaN values exactly consistent with 3 real tiny-lr gradient steps. The actual bug: veRL's
    `save_hf_model_and_tokenizer()` (in `verl/model_merger/base_model_merger.py`) exports a checkpoint
    whose keys **still carry PEFT's wrapper prefix** (`base_model.model.model.*`). Loading that export
    with a plain `Qwen2_5_VLForConditionalGeneration.from_pretrained()` silently drops nearly every real
    weight (name mismatch — HF's loader skips unmatched keys without erroring), leaving the model at
    **near-random initialization**. That one mechanism explained both the gibberish output and the
    apparent vision-weight change. Fixed by reconstructing veRL's *exact* PEFT wrapping
    (`target_modules="all-linear"`, `exclude_modules=".*visual.*"`, matching the real training config)
    and loading the **raw** checkpoint's state dict into that, then `merge_and_unload()`. Final,
    real, decisive verification: **0 missing keys, 0 unexpected keys, all 390 vision parameters
    byte-identical, coherent completions.** This logic now lives in
    `scripts/run_verl_dry_run_on_modal.py::inspect_checkpoint()` — **always load trained veRL+LoRA
    checkpoints this way, never via a plain `from_pretrained()` on the `huggingface/` export.**

**The methodological lesson, if nothing else survives from this log**: when something looks like a
training-correctness failure, check whether it's actually a *loading* failure before trusting the
"training is broken" conclusion. Bug #11 cost the most debugging time and was ultimately not a training
bug at all.

---

## 5. Why veRL, not Unsloth (real comparison, not a hunch)

Investigated seriously after being asked "wouldn't Unsloth just be faster?" Built a real, matched
environment (`configs/modal_app_unsloth.py`) and ran an apples-to-apples comparison
(`scripts/unsloth_smoke_test_2000cap.py` vs. the veRL-proxy numbers in
`scripts/vram_smoke_test_2000cap.py` / `scripts/timing_calibration_2000cap.py`).

**Real result**: Unsloth used dramatically less peak training memory (handled a full 64-sequence batch
in one shot, in less memory than the raw-HF proxy needed for a batch of just 4) — but was **~3.4x
slower overall** in this test (both rollout and per-sequence training compute), the opposite of the
"2x faster" premise that prompted the comparison. Getting Unsloth running at all also required working
around a real, currently **unresolved upstream bug** (`unsloth_zoo`/`vllm` version incompatibility,
confirmed via veRL — no, Unsloth's — own GitHub issue tracker) — a signal that VLM+GRPO is Unsloth's
newer, less battle-tested surface, not its proven one.

Conclusion at the time: stay on veRL, revisit only if veRL's own memory-efficiency levers (see below)
turn out to be insufficient.

**Still open, never tested**: `actor_rollout_ref.model.use_fused_kernels=True` is enabled in the real
training config and is veRL's own documented answer to the same vocab-logits memory bottleneck that
motivated the Unsloth comparison in the first place ("fused cross entropy kernel... avoids materializing
the full logit tensor"). Its real effect on the micro-batch ceiling was never empirically re-measured
after the checkpoint-loading bug (§4 #11) consumed the remaining debugging budget. Worth verifying before
scaling up Phase 1's batch size.

---

## 6. How to actually run things

All commands assume Windows + Git Bash, run from the repo root, with `PYTHONIOENCODING=utf-8` prefixed
(Modal's CLI Unicode output otherwise crashes on Windows console encoding).

```bash
# Cheap environment sanity checks (run these first on a fresh machine)
PYTHONIOENCODING=utf-8 modal run configs/modal_app.py                 # Step 1 smoke test
PYTHONIOENCODING=utf-8 modal run configs/modal_app_verl.py             # veRL environment smoke test

# Local-only sanity checks (no GPU, no Modal, fast)
python scripts/sanity_check_data.py
python scripts/sanity_check_controls.py
python scripts/sanity_check_metrics.py
python scripts/sanity_check_reward_fn.py

# The real training dry run (3 tiny steps, ~10-15 min end to end)
PYTHONIOENCODING=utf-8 modal run scripts/run_verl_dry_run_on_modal.py

# T/D/E sampling harness smoke tests (small scale, real generations)
PYTHONIOENCODING=utf-8 modal run scripts/smoke_test_sampler.py
PYTHONIOENCODING=utf-8 modal run scripts/smoke_test_run_sampling.py
```

`scripts/run_verl_dry_run_on_modal.py` reuses a **persistent Modal Volume**
(`grpo-vlm-verl-checkpoints`) — re-running it overwrites the previous checkpoint. If you want to inspect
an old checkpoint without re-training, use `scripts/inspect_checkpoint_raw.py` or
`scripts/verify_checkpoint_correct_loading.py` directly (both just read the volume, no training).

---

## 7. What's NOT done yet

1. **The real Phase 1 run.** Everything above is Phase 0 (scaffolding + validation) — 3 tiny training
   steps on 50 examples, not the real 300–400 step run PLAN.md describes. Two config values remain
   genuinely undecided before that run can be launched for real (PLAN.md §12):
   - **Prompts-per-training-step** (`train_batch_size` × group size) — the dry run used a
     deliberately tiny value (4 prompts × group size 4); Phase 1 needs a real decision here, which
     directly determines both wall-clock time and cost (PLAN.md §9.2 gives the estimate as a function
     of this, not a fixed number).
   - **Eval-problem-count** for Step 6/7's real evaluation pass — the smoke tests used 2 problems;
     PLAN.md §9.3's cost estimate uses 25 as an illustrative placeholder (matching the headroom check's
     precedent), not a locked decision.
2. **`use_fused_kernels`'s real effect, unverified** — see §5's last paragraph.
3. **Step 6/7's harness has not yet been run at real scale** (n≈128 samples/problem, PLAN.md §4 item 6)
   — only smoke-tested at n=3.
4. **No real Δ_text / Δ_pixel number exists yet.** Everything built so far is infrastructure; the actual
   scientific result (the whole point of PLAN.md) requires the real Phase 1 run plus a real evaluation
   pass on top of it.

---

## 8. File reference

```
PLAN.md                              Research design - read this first for the "why"
IMPLEMENTATION_LOG.md                This file - the "how", and every real bug hit

configs/
  modal_app.py                       Plain HF environment (Steps 1-4, diagnostics)
  modal_app_verl.py                  Real veRL training environment (Step 5+) - has the sed patches, read its docstrings
  modal_app_unsloth.py               Comparison probe only, not used for the real pipeline

src/
  data/                              GSM8K loading, self-rendering, prompts (T/D/E templates)
  controls/                          Manipulation check, truncation check, headroom check
  training/
    prepare_data.py                  GSM8K -> veRL's Parquet schema
    reward_fn.py                     Custom reward (wraps src/metrics/answer_extraction.py)
    dry_run_config.py                The real training CLI args - READ ITS COMMENTS, they document every bug in §4
  inference/
    sampler.py                       Core T/D/E sample generators
    run_sampling.py                  Production orchestration + Parquet persistence
  metrics/
    pass_at_k.py                     Unbiased pass@k estimator
    answer_extraction.py             #### + \boxed{} fallback chain
    bootstrap_ci.py                  Paired bootstrap CI for Delta
    transcription_fidelity.py        Condition D's transcription-accuracy metric

scripts/
  sanity_check_*.py                  Local, no-GPU checks - run these after any src/ change
  run_verl_dry_run_on_modal.py       The real training pipeline orchestrator (Step 5)
  verify_checkpoint_correct_loading.py / inspect_checkpoint_raw.py
                                      Checkpoint-loading diagnostics (§4 bug #11) - use these
                                      patterns for ANY future checkpoint inspection
  smoke_test_sampler.py / smoke_test_run_sampling.py
                                      Step 6 validation
  vram_smoke_test_2000cap*.py / timing_calibration_2000cap.py
                                      Real memory/throughput profiling behind PLAN.md's numbers
  expanded_length_check_tde.py / check_natural_generation_length.py
                                      Real data behind the 1200-token cap decision
  unsloth_smoke_test_2000cap.py      The veRL-vs-Unsloth comparison (§5)
```
