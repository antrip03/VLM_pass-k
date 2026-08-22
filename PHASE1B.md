# Phase 1b — Controls

> ## 🔴 FINDING (2026-08-22): the Phase 1 result is a format-compliance artifact
>
> Step 0 was run on the real records. The measured Δ does not survive
> format-agnostic scoring.
>
> | Condition | Strict scoring (Phase 1) | Format-agnostic scoring |
> |---|---|---|
> | **T** | +0.0587 [+0.041, +0.076] ✅ | **−0.0092** [−0.023, +0.003] ✗ |
> | **D** | +0.0719 [+0.056, +0.089] ✅ | **+0.0002** [−0.013, +0.014] ✗ |
> | **E** | +0.0925 [+0.070, +0.116] ✅ | **−0.0109** [−0.025, +0.004] ✗ |
>
> The sharpening signature goes with it: under fair scoring the
> within-coverage density shift is −0.0099 / +0.0003 / −0.0110 with sign
> tests p = 0.76 / 1.00 / 0.64. **No gain, no sharpening, no expansion.**
>
> **Cause.** `src/training/reward_fn.py` and evaluation share one
> extractor, so RLVR rewarded correctness *and* `####` formatting jointly.
> Base emits a readable answer 87% of the time, RL 96%. Commit `43279ab`
> caught the decoration slice of this (+0.099 → +0.059); the larger slice
> is completions with **no marker at all** — 95.6% of base's unparsed
> condition-T completions — containing correct reasoning written plainly
> ("Therefore, Janet makes **$18** every day...").
>
> **Validation** (`src/metrics/answer_extraction_fallback.py`):
> - Manual review of 120 completions: precision 100% (54/54 per model)
>   — but the extractor was *tuned on those labels*, so this is a
>   training-set figure.
> - Held-out marker-stripped recovery on 3,000 fresh completions per
>   model: 87.4% (base) / 89.0% (RL). The residual error is symmetric and
>   slightly favours RL, i.e. it works *against* the collapse.
> - Base's unparsed completions score like its parsed ones; RL's score far
>   worse — consistent with format compliance, not reasoning.
>
> **Still outstanding:** the 120 labels were produced by an LLM pass that
> also authored the extractor, so they are not independent. A human
> spot-check of ~20 (especially the ambiguous ones) is needed before
> publication. Reproduce with `scripts/rescore_with_fallback.py`.
>
> **What this does to the plan below:** Steps 2–3 (ceiling) and 6
> (paraphrase) now defend a result that no longer exists as stated. Step 4
> (random-reward control) becomes *more* important, and should be run
> under both scorings. The paper's likely reframing is in the closing
> section.

Phase 1b was written to defend Δ_text +0.0587 / Δ_pixel +0.0925 against
reviewer objections. **At a workshop there is no rebuttal round**, so
every objection has to be closed inside the paper.

Nothing here retrains the main model. Everything reuses the existing
checkpoints and records.

---

## What each step defends

| Step | Answers the objection | Cost |
|---|---|---|
| **0** | Multiple — see below | **$0** |
| **1** | "Spurious-reward gains appear fast and plateau; does yours?" | ~$3 |
| **2** | "Is the harder set actually harder?" (gate for step 3) | ~$0.15 |
| **3** | **"Your pass@64 is at 0.98 — 'no expansion' is indistinguishable from 'no room'."** | ~$9 |
| **4** | **"Shao et al. showed Qwen RLVR gains appear with random rewards. Is yours real?"** | ~$13 |
| **6** | "Is RL fragile to *any* distribution shift, not modality specifically?" | ~$1 |

Total ≈ **$26**, against the $31 Phase 1 already cost.

Step 5 (difficulty-matched) turned out to be **free** and moved into
Step 0 — see below.

---

## Step 0 — free, run this first

Everything here is a pure function of records that already exist. No GPU,
no Modal, no credentials.

```bash
python3 scripts/run_step0_analyses.py \
    --base records_base.parquet \
    --rl   records_rl.parquet \
    --out  step0_report.json
```

**Prerequisite:** the two parquet files from the Phase 1 eval. They live
on the Modal account that ran the evaluation:

```bash
modal volume get grpo-vlm-phase1-eval-results records_base.parquet .
modal volume get grpo-vlm-phase1-eval-results records_rl.parquet .
```

What it produces:

1. **Coverage analysis** (`src/analysis/coverage.py`) — the
   ceiling-independent test of Finding 2. Aggregate pass@64 is saturated
   at 0.976–0.990 and both pass@64 deltas are non-significant, so the
   sharpening claim currently rests on a null measured at ceiling. This
   asks the same question of the per-sample records instead: how many
   problems moved from unreachable to reachable (exact McNemar, with an
   explicit `underpowered` flag), does an apparent expansion survive a
   stricter threshold, and — the substantive output — how much
   probability mass moved *within* already-reachable problems.

2. **Residual format confound** (`src/analysis/format_confound.py`) —
   commit `43279ab` removed a 40% inflation, but PLAN.md records that an
   asymmetry remains. This bounds what is left: the **tipping point**
   (what fraction of unparsed completions would have to be secretly
   correct to nullify Δ) against a **measured upper bound** from the
   saved completions. If the bound sits below the tipping point, the
   residual confound demonstrably cannot explain Δ.

3. **Floor effect** (`src/controls/difficulty_matched.py`) — tests the
   caveat PLAN.md already concedes: is Δ_pixel > Δ_text real, or just
   E starting lower? Bins observations by their **own** base rate and
   compares Δ within bins.

4. Truncation symmetry, transcription fidelity, train/eval leakage
   (including the GSM-Plus variants), and a **blinded** spurious-
   correctness review pack.

The review pack needs a human: fill in `verdict` for each trace in
`review_pack/spurious_correctness_review.jsonl`, then score it. Model
identity is held in a separate key file — don't open it first.

---

## Step 1 — trajectory (~$3)

```bash
set -a; source .env; set +a          # HF_TOKEN
modal run scripts/run_trajectory_eval_on_modal.py
# then
modal run scripts/run_trajectory_eval_on_modal.py::analyze_trajectory --labels "base,step_25,step_100,step_200,step_467"
```

Δ_text at steps 25/100/200/467, Condition T only. **A flat curve after
~step 50 is a warning sign.** A climbing curve is reassuring but *not*
conclusive — Shao et al. note random rewards converge slowly (>100 steps
on small models), so this does not replace Step 4.

---

## Steps 2–3 — the ceiling fix (~$9)

**This is the highest-value experiment in the plan.** It is the only
thing that can turn Finding 2 from a measurement-at-saturation into a
real test, and it needs no retraining.

```bash
# Step 2: is the harder set actually harder? (~5 min, base model only)
modal run scripts/run_variant_eval_on_modal.py --dataset gsmplus_harder --mode screen
modal volume get grpo-vlm-phase1b-results gsmplus_harder_screen_summary_base.json .
```

**Decision gate:** look at `per_condition.T.pass_at_64`.
- ≥ 0.95 → still saturated. Do not run Step 3 on this set.
- < 0.85 → ceiling broken, proceed.

```bash
# Step 3: the real run
modal run scripts/run_variant_eval_on_modal.py --dataset gsmplus_harder --mode full
modal run scripts/run_variant_eval_on_modal.py::analyze_variant --tag gsmplus_harder_full --conditions "T,D,E"
```

Uses GSM-Plus (ACL 2024) harder perturbations of **your same 50
problems**, so problem selection is held constant and records pair
against the Phase 1 parquets by `problem_idx`.

⚠️ Harder problems mean longer solutions against a 1200-token cap. Run
the Step 0 truncation comparison on the new records before trusting the
numbers.

---

## Step 4 — random-reward control (~$13)

```bash
modal run scripts/run_random_reward_control_on_modal.py::train_random_reward   # ~4hr A100
modal run scripts/run_random_reward_control_on_modal.py::evaluate              # ~1hr
modal run scripts/run_random_reward_control_on_modal.py::analyze_control
```

Trains to **step 250** with a Bernoulli(0.5) reward and compares against
the real run's **step-250** checkpoint. Step-matched, because comparing
random@250 against real@467 would confound reward-validity with training
length — the asymmetry Shao et al.'s own design avoids.

Verified: `build_args()` for the control differs from the real run in
exactly four keys (reward path, checkpoint dir, step count, experiment
name) and nothing else.

**Plan for the non-null outcome now.** If the random reward reproduces a
large share of the gain, that is a finding, not a failed experiment —
decide how you'd report it before you see the number.

---

## Step 6 — paraphrase control (~$1)

```bash
modal run scripts/run_variant_eval_on_modal.py --dataset gsmplus_rephrased --mode full --conditions T
modal run scripts/run_variant_eval_on_modal.py::analyze_variant --tag gsmplus_rephrased_full --conditions "T"
```

⚠️ **The old `paraphrase_stub()` is not usable for this.** It swapped one
proper noun for another ("Janet" → "Maria") and its own docstring
admitted it was not a real distribution shift. Replaced with GSM-Plus's
`problem understanding` subset — genuine rewrites, answers preserved by
construction, verified on all 50 problems.

Lower marginal value than steps 3 and 4: the "RL is fragile to any
shift" story predicts the gain *shrinks* off-distribution, and yours
*grew* (+9.3% image vs +5.9% text). Cheap enough to be worth having.

---

## Priority if budget is tight

1. **Step 0** — free, and its confidence intervals tell you whether the
   paid steps can even reach a conclusion.
2. **Steps 2–3** — the only thing addressing Finding 2.
3. **Step 4** — closes the best-known critique in this literature.
4. **Steps 1, 6** — cheap, lower marginal value.

Cut Step 6 before cutting Step 3. A reviewer accepts "we didn't run every
control" far more easily than "your central diagnostic was measured at
saturation."

---

## Known gaps, to state in the paper rather than hide

- **Single seed.** CIs are sampling-noise-only (PLAN.md §10 item 8).
- **Temperature not swept.** PLAN.md §4 item 6 asks for a per-model,
  per-k temperature sweep; Phase 1 used a single temperature. Not
  addressed by any step here.
- **The control is at step 250, not 467**, and on the 3B only.
- **Qwen-family monoculture.** Shao et al.'s warning — "RLVR research
  validated solely on Qwen models might not generalize" — applies. Step 4
  addresses it by direct control rather than by cross-family replication,
  which the budget precluded. Note that InternVL/Ovis would *not* have
  fixed this: their LLM backbones are also Qwen.

---

## Verification

```bash
python3 scripts/smoke_test_step0_analyses.py
```

40+ assertions against hand-computed answers on synthetic data. It caught
a real precision bug during development, and it verifies that the
floor-effect control separates two worlds which produce *identical*
misleading aggregate signatures.
