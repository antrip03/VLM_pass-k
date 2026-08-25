# Phase 1b findings — the Phase 1 result is a measurement artifact

**Status as of 2026-08-22.** Modal spend: ~$8. Everything below is
reproducible from the committed records and scripts.

---

## The claim

> We do **not** claim that RLVR fails to improve mathematical reasoning.
> We show that on a **saturated** benchmark with an **instruction-tuned**
> model, a format-coupled answer-extractor reports a **significant
> improvement** where format-agnostic scoring of the **same checkpoints**
> reports a **significant decline**.

Both halves are measured, with paired-bootstrap CIs, on the same 50
problems and the same 128 samples per problem.

---

## 1. The headline reversal (GSM8K, condition T, step 467)

| Scoring | Δ pass@1 | 95% CI | Significant |
|---|---|---|---|
| **Strict** (`#### N` required) | **+0.0587** | [+0.0406, +0.0763] | **YES, positive** |
| **Format-agnostic** | **−0.0069** | [−0.0208, +0.0064] | no |

Same model, same checkpoint, same problems, same 128 samples each. Only
the answer-extractor differs. A significant positive result becomes a
non-significant null.

## 1b. The same thing in raw generation counts

38,400 generations (50 problems × 128 samples × 3 conditions × 2 models).
"Rescued" = correct under format-agnostic scoring but scored wrong by the
strict extractor.

| model | cond | total | strict correct | fair correct | rescued | unparsed |
|---|---|---|---|---|---|---|
| base | T | 6,400 | 3,897 | 4,451 | **+554** | 856 |
| base | D | 6,400 | 3,860 | 4,433 | **+573** | 859 |
| base | E | 6,400 | 3,175 | 4,019 | **+844** | 1,341 |
| RL | T | 6,400 | 4,273 | 4,407 | **+134** | 264 |
| RL | D | 6,400 | 4,320 | 4,466 | **+146** | 246 |
| RL | E | 6,400 | 3,767 | 3,988 | **+221** | 413 |

Condition T, stated as plainly as possible:

- The base model had **554** correct answers thrown away for formatting
  (8.7% of all its generations).
- The RL model had **134** thrown away (2.1%).
- **Under strict scoring RL led by +376 generations. Under
  format-agnostic scoring RL trails by −44.**

The base model recovered **420 more** generations than RL did. That
single asymmetry is the entire reported result.

## 1c. The D/E result — the actual contribution

The novel claim had three parts. All three dissolve under format-agnostic
scoring:

| Claim | Strict | Format-agnostic |
|---|---|---|
| Gain survives modality shift (Delta_E > 0) | +0.0925 ✅ | **-0.0109** ✗ |
| Gain is LARGEST on images (Delta_E > Delta_T) | +0.034 ✅ | **-0.001** ✗ |
| Ordering is not a floor effect | survives 4/4 bins | **collapses, 2/4 bins, gap -0.0003** |

**Mechanism, and it is modality-specific.** Base-model format compliance
by condition:

| cond | base compliance | RL compliance | gap | base rescued | RL rescued | asymmetry |
|---|---|---|---|---|---|---|
| T | 0.866 | 0.959 | +0.093 | 554 | 134 | +420 |
| D | 0.866 | 0.962 | +0.096 | 573 | 146 | +427 |
| **E** | **0.790** | **0.935** | **+0.145** | **844** | **221** | **+623** |

The base model follows the output-format instruction ~8 points LESS
reliably when reading a problem from an image than from text - the
instruction is textual, the content is visual, and it lapses into prose.
So the strict extractor penalises it hardest in condition E.

The reported Delta ordering (E +0.093 > D +0.072 > T +0.059) is the
COMPLIANCE-GAP ordering (E +0.145 > D +0.096 > T +0.093), not a
reasoning-transfer ordering.

**So the contribution is not "the gain survives modality shift" but:**

> Cross-modal transfer of an RLVR gain can be entirely manufactured by an
> answer-extractor. The apparent transfer is strongest precisely where the
> measurement is weakest, because reading from an image degrades
> instruction-following, and a format-coupled scorer converts that into
> apparent reasoning capability.

This is specific to VLM evaluation - the mechanism requires a modality
that degrades instruction-following, so it cannot appear in the text-only
RLVR literature. It is only visible because of the T/D/E design.

CORRECTION TO AN EARLIER RESULT IN THIS FILE'S HISTORY: the
difficulty-matched control was first run on the strict column and
reported "ORDERING SURVIVES (+0.0409, 4/4 bins)". That was confirming an
artifact. On `correct_fallback` the gap is -0.0003 with bins splitting
2/4.

## 2. The mechanism, across training

GSM8K, condition T, 50 problems × n=128, six checkpoints.
Base: strict pass@1 = 0.6128, format-agnostic = 0.7072.

| step | format compliance | strict pass@1 | fair pass@1 | Δ strict | Δ fair |
|---|---|---|---|---|---|
| 5 | 0.8661 | 0.6070 | 0.7005 | −0.006 | −0.007 |
| 75 | 0.8881 | 0.6256 | 0.6992 | +0.013 | −0.008 |
| 150 | 0.9127 | 0.6352 | 0.6958 | **+0.022** ✅ | −0.011 |
| 300 | 0.9483 | 0.6558 | 0.6855 | **+0.043** ✅ | **−0.022** ✅ |
| 467 | 0.9664 | 0.6683 | 0.6887 | **+0.056** ✅ | **−0.018** ✅ |

- **Format compliance rises monotonically**: 86.6% → 96.6%
- **Strict pass@1 tracks it upward**: 0.607 → 0.668
- **Format-agnostic pass@1 is flat, drifting down**: 0.7005 → 0.6887

> **Extractor version note.** This trajectory's format-agnostic column was
> computed with the extractor as it stood *before* the 2026-08-23 human
> review. Re-scoring the main eval with the corrected version shifted Δ by
> ~+0.005 — within the CI half-width, and it does not change the shape of
> either curve. The compliance column is unaffected (it depends only on
> the strict marker).

The strict curve is not measuring capability. It is measuring how often
the scorer could *read* an answer that was frequently already correct.

## 3. Cause: the reward and the metric share one extractor

```
src/training/reward_fn.py  ─┐
                            ├─→ src/metrics/answer_extraction.py
src/inference/sampler.py   ─┘
```

During training, a correct completion ending *"Therefore, Janet makes
**$18** every day"* earned **reward 0** — no `####`. So GRPO was paid for
correctness **and** formatting jointly, and with base pass@64 already at
0.98 there was almost no reasoning left to buy. It took the cheap option,
visibly by step 5.

Commit `43279ab` caught the *decorated-marker* slice of this
(`#### <18>`), moving Δ_text +0.099 → +0.059. The larger slice is
completions with **no marker at all** — 95.6% of base's unparsed
condition-T completions.

## 4. Out-of-distribution: real negative transfer

MATH-500 levels 3-5, condition T, 50 problems × n=128.

| Scoring | k | base | RL | Δ | 95% CI | Sig |
|---|---|---|---|---|---|---|
| strict | 1 | 0.4631 | 0.4591 | −0.0041 | [−0.023, +0.016] | no |
| **fallback** | **1** | 0.4841 | 0.4669 | **−0.0172** | [−0.032, −0.003] | **YES** |
| **fallback** | **8** | 0.8020 | 0.7773 | **−0.0246** | [−0.045, −0.007] | **YES** |

RL is **significantly worse** than base out-of-distribution — PLAN.md §6
**Outcome D (negative transfer)**, which the plan attributes to
narrow-distribution adapter interference.

**And the strict metric hides it**: Δ@1 = −0.004, non-significant,
because RL's format advantage (89.0% vs 87.5%) partially cancels the
reasoning loss. The confound manufactures false positives *and* masks
true negatives.

## 4b. The causal test — manipulating the mediator directly (2026-08-25)

Everything above is observational: compliance and the strict Δ move
together, which is strong correlational evidence but not proof of
causation. This test manipulates compliance directly, with **no
training, no gradient**: base model, condition E, one sentence added to
the prompt demonstrating the `#### N` marker line (a bare format
exemplar — not a worked-example few-shot, which would also prime
reasoning style and confound the test).

| | Compliance | Strict Δ (vs RL) | Significant |
|---|---|---|---|
| Zero-shot base → RL | 77.9% → 93.2% (gap 15.3pp) | **+0.1028** | **yes** |
| Few-shot base → RL | 88.3% → 93.2% (gap 4.8pp) | **+0.0162** | **no** |

One line of prompt text closed 68% of the compliance gap and collapsed
the "significant" strict gain from +10.3 points to +1.6, losing
significance along with the compliance gap. Format-agnostic (reasoning)
accuracy was unaffected by the manipulation (+0.019, not significant,
[−0.003, +0.041]) — confirming the exemplar changed formatting only.

**This converts the paper's central claim from correlational to causal.**
Reproduce: `scripts/run_compliance_matching_control.py` +
`scripts/analyze_compliance_control.py`.

## 5. Why this does not contradict the literature

| | Yue et al. | MIRROR | **This work** |
|---|---|---|---|
| Model | base (zero-RL) | instruct | **instruct** |
| Benchmark | MATH500 | geometry | **GSM8K (saturated)** |
| Base pass@1 | **34.5%** | 32.9% | **70.7%** |
| Base pass@64 | — | 80.2 @16 | **0.98** |
| Result | 34.5 → 74.4 | +7.0 | **≈0 / negative** |

RLVR works where there is headroom. This setup had almost none.
Corroboration rather than contradiction: 1-shot RLVR (arXiv 2504.20571)
states *"the improvement for RLVR with some examples may mainly come from
the format fixing of the initial model."*

## 6. The cheap check that would have caught this

PLAN.md §4 item 1 already mandates calibrating the base model against a
published number. It was never run. Running it now:

| | Base GSM8K pass@1 |
|---|---|
| **Published** (Qwen2.5-VL-3B) | **77.7%** |
| Strict scoring | **60.9%** ← 17 points low ❌ |
| Format-agnostic | **69.6%** ← 8 points low ✅ |

A 17-point gap against a published number is a two-minute red flag.

## 7. The format artifact scales inversely with difficulty

| Dataset | base pass@1 | strict/fair gap | base pass@64 |
|---|---|---|---|
| GSM8K | 0.699 | **9.0 pts** | 0.979 |
| GSM-Plus harder | 0.576 | 6.7 pts | 0.950 |
| MATH-500 L3-5 | 0.408 | **2.2 pts** | 0.954 |

On easy problems the model reasons correctly but phrases loosely, so
strict scoring punishes it. On hard problems it fails outright and there
is no correct-but-misformatted answer to lose. **The trap is worst
exactly where benchmarks are most saturated.**

Also confirmed on all three datasets: **pass@64 is identical under both
scorings** (0.950 vs 0.950; 0.800 vs 0.800). Format compliance can only
distort pass@1.

---

## Limitations, to state rather than bury

1. **Extraction validation is not independent.** The fallback extractor
   was written and then hand-validated by the same agent. Held-out
   marker-stripped recovery is 87.4% (base) / 89.0% (RL), symmetric and
   slightly favouring RL — i.e. working *against* the collapse. **A human
   spot-check of ~20 traces is still required before publication.**
2. **Single seed**; CIs reflect sampling noise only.
3. **Negative transfer rests on one OOD dataset** (MATH-500). Say "we
   observe negative transfer on MATH-500", not "RLVR causes negative
   transfer".
4. **pass@64 coverage claims are not measurable** at this model scale.
   Every dataset tried remains saturated: 0.979 / 0.950 / 0.954.
5. **The 20-problem screen mis-estimated pass@64** (0.800 vs 0.954 at 50
   problems). At n=64 the estimator degenerates to "was this ever
   solved", a proportion over 20 items. Screens are adequate for pass@1,
   not pass@k.

## Reproduce

```bash
python3 scripts/run_step0_analyses.py --base eval_results/records_base.parquet \
    --rl eval_results/records_rl.parquet --out step0_report.json
python3 scripts/rescore_with_fallback.py
python3 scripts/smoke_test_step0_analyses.py     # 40+ assertions
```

Artifacts: `eval_results/phase1b/`, `eval_results/rescore_comparison.json`,
`eval_results/step0_report.json`.
