# Phase 1 + 1b — Complete summary for the workshop paper

**For the teammate.** Everything below is measured and reproducible from
this branch. Total Modal spend: Phase 1 ~$31, Phase 1b ~$19.

---

## 1. What we set out to test

Does the pass@1/pass@k "sharpening" signature of RLVR (Yue et al., arXiv
2504.13837) survive a **text → image modality shift**?

Design (PLAN.md §1.7): train Qwen2.5-VL-3B-Instruct with Dr. GRPO on
**text-only** GSM8K, freeze the vision tower, then evaluate the same
problems in three conditions:

| | |
|---|---|
| **T** | problem as plain text |
| **D** | problem as image → model transcribes → solves its own transcription |
| **E** | problem as image → solve end-to-end |

D and E are the novel part. Nobody had tested whether a text-trained RLVR
gain transfers to pixel input on matched content.

Training: 467 steps (one clean epoch, 7473/16), LoRA r=32 on the decoder
only, vision tower excluded. Vision weights verified **byte-identical**
before/after (sha256 `48ed880e…163cf`), so any measured effect is in the
reasoning pathway, not perception.

## 2. What we first measured

| Condition | Δ pass@1 | 95% CI | Sig |
|---|---|---|---|
| T (text) | +0.0587 | [+0.041, +0.076] | ✅ |
| D | +0.0719 | [+0.056, +0.089] | ✅ |
| **E (image)** | **+0.0925** | [+0.070, +0.116] | ✅ |

This looked like a clean, publishable result: the gain survives the
modality shift and is *largest* on images. pass@64 flat at 0.976–0.990 —
the textbook sharpening signature.

## 3. What went wrong, and how we found it

PLAN.md §4 item 5 mandates a manual extraction spot-check. Running it
revealed that the answer-extractor was discarding correct answers.

**The scoring rule required the model to end with `#### N`** — GSM8K's own
ground-truth convention, requested in the prompt. Completions like

> "Therefore, Janet makes **$18** every day at the farmers' market."

were scored **wrong**: correct reasoning, unreadable to the scorer.

**The critical detail:** `src/training/reward_fn.py` and
`src/inference/sampler.py` import the *same* extractor. So RLVR was paid
for correctness **and** formatting jointly. With base pass@64 already at
0.98, there was almost no reasoning left to buy — so the model bought
formatting.

Commit `43279ab` caught a decorated-marker slice of this (`#### <18>`,
itself induced by the prompt's own `<number>` placeholder) and moved
Δ_text from +0.099 → +0.059. But **95.6% of unparsed base completions had
no marker at all** — an instruction-following failure, not an extractor
bug, and not addressed by that fix.

## 4. Corrected results — GSM8K

Same records, same samples, rescored with a format-agnostic extractor
(`src/metrics/answer_extraction_fallback.py`).

| Condition | Strict Δ pass@1 | Format-agnostic Δ pass@1 |
|---|---|---|
| **E (image)** | **+0.0925** [+0.070, +0.116] ✅ | **-0.0048** [-0.020, +0.011] ✗ |
| **D** | **+0.0719** [+0.056, +0.089] ✅ | **+0.0052** [-0.009, +0.019] ✗ |
| T (control) | **+0.0587** [+0.041, +0.076] ✅ | **-0.0069** [-0.021, +0.006] ✗ |

**In raw counts** (6,400 generations per model per condition). "Rescued" =
correct under fair scoring, discarded by strict:

| model | cond | strict ✓ | fair ✓ | rescued | unparsed |
|---|---|---|---|---|---|
| base | E | 3,175 | 4,019 | **+844** | 1,341 |
| RL | E | 3,767 | 3,988 | +221 | 413 |
| base | T | 3,897 | 4,451 | +554 | 856 |
| RL | T | 4,273 | 4,407 | +134 | 264 |

Condition T in one line: under strict scoring RL led by **+376
generations**; under fair scoring it **trails by −44**.

## 5. The mechanism is modality-dependent — the novel finding

Format compliance (fraction of generations the scorer could read):

| cond | base | RL | **gap** |
|---|---|---|---|
| T | 0.866 | 0.959 | +0.093 |
| D | 0.866 | 0.962 | +0.096 |
| **E** | **0.790** | 0.935 | **+0.145** |

**The base model follows the output-format instruction ~8 points less
reliably when reading from an image than from text.** The instruction is
textual, the content is visual, and it lapses into prose more often. So a
format-coupled scorer penalises it hardest in condition E.

The reported Δ ordering (E +0.093 > D +0.072 > T +0.059) is the
**compliance-gap ordering** (E +0.145 > D +0.096 > T +0.093).

A difficulty-matched control confirms it: under strict scoring the
Δ_E > Δ_T ordering "survives" difficulty matching (+0.0409, 4/4 bins);
under fair scoring it collapses (**−0.0003, 2/4 bins**).

## 6. The mechanism over training

GSM8K, condition T, six checkpoints.

| step | format compliance | strict pass@1 | fair pass@1 | Δ strict | Δ fair |
|---|---|---|---|---|---|
| base | 0.866 | 0.6128 | 0.7072 | — | — |
| 5 | 0.8661 | 0.6070 | 0.7005 | −0.006 | −0.007 |
| 75 | 0.8881 | 0.6256 | 0.6992 | +0.013 | −0.008 |
| 150 | 0.9127 | 0.6352 | 0.6958 | **+0.022** ✅ | −0.011 |
| 300 | 0.9483 | 0.6558 | 0.6855 | **+0.043** ✅ | **−0.022** ✅ |
| 467 | 0.9664 | 0.6683 | 0.6887 | **+0.056** ✅ | **−0.018** ✅ |

**At step 467 the two scoring rules reach opposite, statistically
significant conclusions on the identical checkpoint.** Compliance climbs
monotonically 86.6% → 96.6%; format-agnostic accuracy is flat and drifts
slightly down.

## 7. Robustness — GSM-Plus harder (Run A)

Harder perturbations of the same 50 problems (12 points more pass@1
headroom), 50 × n=128, all three conditions.

| Condition | Scoring | k | base | RL | Δ | 95% CI | Sig |
|---|---|---|---|---|---|---|---|
| **E** | strict | 1 | 0.3984 | 0.4739 | **+0.0755** | [+0.056, +0.095] | ✅ |
| **E** | fallback | 1 | 0.4948 | 0.5034 | **+0.0086** | [−0.003, +0.021] | ✗ |
| **E** | fallback | 64 | 0.9127 | 0.8989 | −0.0138 | [−0.058, +0.026] | ✗ |
| **D** | strict | 1 | 0.4864 | 0.5359 | **+0.0495** | [+0.035, +0.064] | ✅ |
| **D** | fallback | 1 | 0.5525 | 0.5544 | **+0.0019** | [−0.007, +0.011] | ✗ |
| **D** | fallback | 64 | 0.8819 | 0.8976 | +0.0157 | [−0.004, +0.043] | ✗ |
| T | strict | 1 | 0.4856 | 0.5342 | **+0.0486** | [+0.034, +0.064] | ✅ |
| T | fallback | 1 | 0.5491 | 0.5517 | **+0.0027** | [−0.009, +0.016] | ✗ |
| T | fallback | 64 | 0.9096 | 0.9053 | −0.0043 | [−0.042, +0.031] | ✗ |

Fraction of the strict gain explained by formatting: **E 89%, D 96%,
T 94%.** E again shows the largest fake gain, as the compliance mechanism
predicts.

**This rules out the strongest counter-explanation:** the null is not an
artifact of GSM8K saturation. The same pattern appears with real headroom.

## 8. Out-of-distribution — MATH-500 (condition T only)

| Scoring | k | base | RL | Δ | 95% CI | Sig |
|---|---|---|---|---|---|---|
| strict | 1 | 0.4631 | 0.4591 | −0.0041 | [−0.023, +0.016] | ✗ |
| **fallback** | 1 | 0.4841 | 0.4669 | **−0.0172** | [−0.032, −0.003] | **✅ negative** |
| **fallback** | 8 | 0.8020 | 0.7773 | **−0.0246** | [−0.045, −0.007] | **✅ negative** |

RL is **significantly worse** than base out-of-distribution — PLAN.md §6
**Outcome D (negative transfer)**. And strict scoring *hides* it (Δ@1 =
−0.004, n.s.) because RL's format advantage partially cancels the
reasoning loss.

**The confound manufactures false positives and masks true negatives.**

D/E were not run on MATH-500: its problems contain LaTeX and Asymptote
diagram source, which `render.py` would draw as raw markup rather than
typeset maths — a different task from "same content, different
representation."

## 9. Why this does not contradict the RLVR literature

| | Yue et al. | MIRROR | **This work** |
|---|---|---|---|
| Model | base (zero-RL) | instruct | **instruct** |
| Benchmark | MATH500 | geometry | **GSM8K (saturated)** |
| Base pass@1 | **34.5%** | 32.9% | **70.7%** |
| Result | 34.5 → 74.4 | +7.0 | **≈0** |

RLVR reliably works **where there is headroom**. This setup had almost
none — 98% of problems already solvable within 64 attempts. The reward
offered two routes to a higher score; only one was cheap.

Corroboration rather than contradiction — 1-shot RLVR (arXiv 2504.20571):
> "the improvement for RLVR with some examples may mainly come from the
> **format fixing** of the initial model."

We quantify what they asserted, and show it is **modality-dependent**.

## 10. The two-minute check that would have caught this

PLAN.md §4 item 1 already mandates calibrating the base model against a
published number. It was never run.

| | Base GSM8K pass@1 |
|---|---|
| **Published** (Qwen2.5-VL-3B) | **77.7%** |
| Our strict scoring | **60.9%** ← 17 points low ❌ |
| Our format-agnostic scoring | **69.9%** ← 8 points low ✅ |

Also: **the artifact scales inversely with difficulty** (strict/fair gap
9.0 → 6.7 → 2.2 points across GSM8K → GSM-Plus → MATH-500). It bites
hardest exactly where benchmarks are most saturated. And **pass@64 is
identical under both scorings on every dataset** — format compliance can
only ever distort pass@1.

---

## 11. Suggested paper flow

1. **Introduction** — RLVR, the sharpening hypothesis, and the untested
   modality-shift question. State the T/D/E design as the contribution's
   enabling apparatus.
2. **Setup** — model, Dr. GRPO, 467 steps, vision-tower freeze +
   byte-identity check, T/D/E conditions, pass@k estimator.
3. **First result** (§2 above) — present it as it was measured: a
   significant gain that survives modality shift and is largest on
   images. Do not signpost the twist yet.
4. **The extraction spot-check** (§3) — a mandated control turns up a
   scoring failure. Trace it to the shared reward/eval extractor.
5. **Corrected results** (§4) — everything dissolves. Give raw counts;
   they are more persuasive than deltas.
6. **Why the image condition was worst** (§5) — the modality-dependent
   compliance gap. **This is the novel contribution.**
7. **The mechanism over training** (§6) — the divergence figure. Two
   scoring rules, opposite significant conclusions.
8. **Robustness** (§7, §8) — GSM-Plus harder rules out saturation;
   MATH-500 shows the confound also masks a true negative.
9. **Relation to prior work** (§9) — explicitly *not* a contradiction.
10. **Recommendations** (§10) — report both scorings; calibrate against a
    published number; never share an extractor between reward and metric.
11. **Limitations** (§12).

## 12. What we can and cannot claim

**Can claim, fully supported:**
- On this model and benchmark, a format-coupled extractor reports a
  significant gain where format-agnostic scoring of the same checkpoints
  reports no gain (GSM8K, GSM-Plus) or a significant decline (MATH-500).
- The artifact is **larger in image conditions**, because reading from an
  image degrades instruction-following. This is VLM-specific and cannot
  appear in the text-only RLVR literature.
- The apparent cross-modal transfer, its magnitude ordering, and the
  difficulty-matched defence of that ordering all dissolve under
  format-agnostic scoring.

**Must NOT claim:**
- ❌ "RLVR does not improve mathematical reasoning." Contradicted by Yue
  et al. and others; our regime had no headroom.
- ❌ "RLVR causes negative transfer." One OOD dataset. Say *"we observe
  negative transfer on MATH-500."*
- ❌ Any coverage/expansion claim from pass@64. Every dataset remained
  near-saturated (0.88–0.98).
- ❌ That the effect generalises across models. Single model, single seed.

**Safe framing for the abstract:**
> We do not claim RLVR fails to improve mathematical reasoning. We show
> that when an RLVR reward function and its evaluation metric share an
> answer-extractor, format compliance learned during training is measured
> as reasoning capability — and that in a vision-language setting this
> artifact is *largest under image input*, because reading from an image
> degrades instruction-following.

## 13. Limitations to state, not bury

1. **Extraction validation is not fully independent.** The fallback
   extractor was written and then hand-validated by the same agent.
   Held-out marker-stripped recovery: **87.4% (base) / 89.0% (RL)** —
   symmetric and slightly favouring RL, i.e. biased *against* our
   conclusion. **A human spot-check of ~20 traces remains outstanding**
   (`eval_results/extraction_review/`).
2. **Single seed**; CIs reflect sampling noise only, not seed variance.
3. **50 evaluation problems** — a budget-driven choice (PLAN.md §9.3).
4. **pass@64 coverage claims are not measurable** here: 0.979 / 0.950 /
   0.954 / 0.88–0.91 across datasets.
5. **Negative transfer rests on one OOD dataset.**
6. **MATH-500 was condition-T only** (LaTeX/diagram source does not
   render as prose).
7. **A 20-problem screen mis-estimated pass@64** (0.800 vs 0.954 at 50
   problems). At n=64 the estimator degenerates to a yes/no over 20
   items. Screens are adequate for pass@1, not pass@k.

## 14. Reproduce

```bash
python3 scripts/run_step0_analyses.py \
    --base eval_results/records_base.parquet \
    --rl   eval_results/records_rl.parquet --out step0_report.json
python3 scripts/rescore_with_fallback.py
python3 scripts/smoke_test_step0_analyses.py    # 40+ assertions
```

Artifacts: `eval_results/` (Phase 1 records, 38,400 generations with full
completion text), `eval_results/phase1b/` (trajectory, MATH-500,
GSM-Plus). Detail in `FINDINGS.md`; run instructions in `PHASE1B.md`.
