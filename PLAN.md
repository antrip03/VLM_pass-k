# Master Plan: Does RLVR's Reasoning Gain Survive a Modality Shift?

**Testing the sharpening hypothesis on VLM math reasoning via text-to-pixel rendering, using GRPO-family RL and pass@1/pass@k diagnostics.**

**Status:** Pre-implementation master plan. Nothing has been trained yet. This document is meant to be shared with the team as the single source of truth before writing any code.

---

## 0. One-Paragraph Summary

We take a vision-language model (VLM), fine-tune it with RL (Dr. GRPO) using **only text** math problems, and then test whether the improvement this RL training produces **survives being shown the identical problem as a picture instead of typed text**. If the improvement collapses in picture mode, that suggests RL taught the model a shortcut tied to the text format it trained on ("sharpening"). If the improvement survives, that suggests RL taught something more general. We measure this using **pass@1** (first-try accuracy) and **pass@k** (accuracy given k tries) — a diagnostic already used for text-only models, never yet applied to this specific text→image transfer question. Every outcome is publishable; the design does not depend on getting a particular answer.

---

## 1. The Whole Idea, Explained Simply

### 1.1 What RL / GRPO training does

Two ways to train a math-solving AI: **supervised training** (show it worked examples, say "copy this"), or **RL** (give it a problem, let it try, only say "right"/"wrong"). **GRPO** (Group Relative Policy Optimization) is a specific, popular RL method: the model generates a *group* of attempts at the same problem; attempts that scored above the group's average get nudged more likely, attempts below get nudged less likely. After RL training, accuracy goes up. Nobody disputes that.

### 1.2 The mystery: why does it go up?

- **Story 1 — "Sharpening"**: the model already secretly "knew" the answer — given many guesses, it would eventually land on it. RL just made it state that answer more confidently on the *first* try. Nothing new learned, old knowledge promoted.
- **Story 2 — "New capability"**: the model genuinely learned a method it didn't have before — something unreachable even with many guesses before training.

Both stories can produce the same accuracy increase. You need a cleverer test to tell them apart.

### 1.3 The clever test: pass@1 vs. pass@k

Treat each question to the model like buying a lottery ticket (models have built-in randomness, so asking twice can give different answers):

- **pass@1** = buy 1 ticket. Win? Averaged over many problems = ordinary accuracy.
- **pass@k** (e.g. pass@8, pass@64) = buy k tickets. Did **at least one** win?

pass@1 measures **precision** (is the right answer the favorite guess?). pass@k, as k grows, measures **coverage** (can the model ever produce the right answer, given enough tries?).

- **Sharpening predicts:** pass@1 ↑, pass@k stays flat (or dips slightly — concentrating on one answer can crowd out rare-but-correct alternatives).
- **New capability predicts:** pass@1 ↑ **and** pass@k ↑ — genuinely new winning tickets were added.

**This part already exists** (Yue et al., arXiv 2504.13837) — but only for text-only models. No pictures involved. Not our novelty.

### 1.4 Our twist: what if we change how the question is shown?

RL training here only ever shows the model **typed text** math problems — never pictures. So: maybe whatever "shortcut" RL teaches is glued to the *text* format specifically. Test: take the identical problem, show it two ways — typed text, and a rendered **picture** of that same problem. Run both the untrained ("base") model and the RL-trained model on both versions.

Define:
- **Δ_text** = (RL model's text accuracy) − (base model's text accuracy) — how much RL helped, in the format it trained on.
- **Δ_pixel** = (RL model's image accuracy) − (base model's image accuracy) — how much RL helped, when shown a picture instead.

Compare Δ_text to Δ_pixel. **That comparison is the whole experiment.**

### 1.5 Why we still need pass@k here — the "perception tax" trap

Pictures are naturally harder to read than typed text. If scores drop in picture mode, is it because (a) the model can't correctly *see* the numbers, or (b) it reasons worse once it has read them correctly? pass@k answers this: if pass@k stays similar between text and picture mode (for both models), the content mostly *did* get through — ruling out "it just can't read the picture" as the explanation for whatever happens to pass@1. This cost of imperfect picture-reading, paid regardless of anything else, is the **perception tax**.

### 1.6 What "the shortcut" concretely means (documented, not hypothetical)

A shortcut = the model adopts a behavior because it correlated with reward during training, not because it reflects real understanding. Real documented examples:
1. **Length gaming**: GRPO's reward math can make longer wrong answers get under-penalized and short right answers over-rewarded — models drift toward exploiting this, unrelated to actually solving problems better.
2. **Hollow self-checking**: RL-trained models say "let me verify..." far more often, but research shows these "rechecks" are mostly confirmatory restatements, not real re-derivation, and often don't improve accuracy.
3. **Bare-answer shortcutting**: reasoning explanations can collapse to near-nothing if only the final answer is checked.

**Honest complication**: this isn't one-directional — one study (Logic-RL) found RL-trained models were *more* robust to superficial changes (renamed variables) than supervised-trained models. So "RL = shallow shortcuts" is not a universal law — it's the specific hypothesis this experiment tests, not an assumed fact.

### 1.7 The upgraded three-condition design

Instead of only Text vs. Image, we add a **third condition** that decomposes perception from reasoning:

- **Condition T (Text)**: problem as typed text.
- **Condition D (Decomposed)**: show the image, ask the model to **transcribe only** (no solving), then feed that transcription back as text and ask it to solve.
- **Condition E (End-to-end)**: show the image, ask the model to perceive *and* reason in one continuous pass.

This gives us:
- **T vs. D** — isolates pure transcription fidelity (does self-reading introduce errors?). Also gives a directly reportable metric: transcription accuracy.
- **D vs. E** — isolates what's lost specifically from doing perception+reasoning *jointly* vs. as separate explicit steps.
- **Δ_RL measured at all three conditions** — tells us *where in the pipeline* RL's advantage disappears, not just *that* it disappears.

Condition D costs roughly 2 generation calls per sample (transcribe, then solve) instead of 1 — total eval cost across all three conditions ≈ 1.6–2× the original two-condition estimate.

---

## 2. The Hypothesis

### 2.1 Definitions

- π_base = the base (pre-RL) VLM.
- π_RL = π_base + a LoRA fine-tune using **Dr. GRPO** (see §7), trained on **text-only** math problems. Vision encoder + projector are **frozen** — never updated.
- Δ_text = pass@1(π_RL, T) − pass@1(π_base, T)
- Δ_pixel = pass@1(π_RL, E) − pass@1(π_base, E) (also computed at Condition D)
- pass@k = unbiased estimator: sample n ≫ k completions, pass@k = 1 − C(n−c, k)/C(n, k), where c = number correct.

### 2.2 Primary hypothesis

**H-representation-bound**: Δ_pixel is significantly smaller than Δ_text (via bootstrap/Bayesian CI on the difference), **and** pass@k(π_RL) is not significantly different from pass@k(π_base) in any of the three conditions.

Framed as *measuring a quantity* (how much of Δ_text survives as Δ_pixel), not a yes/no bet — every real outcome is a point on this same scale.

### 2.3 The pass@k hypothesis, specifically — two parts

- **(a) Expected/calibration** (not novel on its own): pass@k(π_RL, T) ≈ pass@k(π_base, T) — replicates the known text-only sharpening signature. If this fails, the checkpoints are broken; stop.
- **(b) Genuinely new**: pass@k(π_RL, E) ≈ pass@k(π_base, E) — does that same "flat coverage" pattern *also* hold under a modality shift? Nobody has tested this (verified — see §5).

---

## 3. Implementation Plan

### Phase 0 — Smoke Test / Proof-of-Concept (3B model, small scale)

**Goal**: confirm the whole pipeline works end-to-end before spending real compute. Not a scientific result — a plumbing check.

1. Model: **Qwen2.5-VL-3B-Instruct**, downloaded as-is (already trained by Alibaba; we do not train a VLM from scratch).
2. **Cold-start / headroom pre-check**: sample pass@1 and pass@16 on ~30–50 text problems using the untrained base model. Confirm a real gap exists (pass@16 clearly higher than pass@1) — this means there's "latent capability" for RL to plausibly sharpen. If pass@16 ≈ pass@1 ≈ near-zero, the model/dataset pairing won't show a usable signal; stop and reconsider before any training.
3. **Manipulation-check dry run**: confirm LoRA `target_modules` correctly excludes the vision tower and projector — diff the relevant state-dict keys before touching training.
4. Run a **tiny GRPO/Dr. GRPO training pass**: ~20–30 steps, small group size, just to confirm the training loop runs, rewards compute correctly, and loss moves in a sane direction.
5. Run a **tiny eval pass**: ~10 problems, n=5 samples, both text and image mode, confirm generation, answer extraction (regex on GSM8K's `#### number` format), and pass@k computation all work without errors.
6. **Timing calibration**: measure real tokens/sec on the actual hardware being used — do not trust the estimates in §9 blindly; use them as priors, confirm with ~15 minutes of real measurement.
7. **Truncation check** at the 1200-token cap: confirm truncation rate is low (<5%) on this small sample before scaling up.

**Exit criterion**: all of the above run cleanly, pass@16 > pass@1 by a meaningful margin on text mode, and timing roughly matches the estimates in §9. Only then proceed to Phase 1.

### Phase 1 — Full Run (3B model)

1. **Algorithm**: Dr. GRPO (not vanilla GRPO, not CISPO — see §7 for why).
2. **Training data**: text-only math problems (GSM8K training split or similar), never images.
3. **LoRA**: adapters on LLM decoder layers only; vision encoder + projector excluded from `target_modules` and frozen.
4. **Seeds**: 1 (single training run, seed 0). A single-seed design trades away characterizing seed-to-seed variance (a real, documented effect in RLVR training — seed changes alone can shift pass@1 by several points on comparably-sized benchmarks) for a much smaller compute/time footprint. Deliberate scope decision, recorded explicitly as a named limitation in §10 item 8.
5. **Batch shape**: group size = 8, prompts-per-training-step = 16 → S = 128 total sequences/step. Group size matches the real Dr. GRPO recipe's per-GPU share (`sail-sg/understand-r1-zero/train_zero_math.py`: `rollout_batch_size=128` prompts globally over 8 GPUs = 16 prompts/GPU, `num_samples=8`). Prompts-per-step chosen at the cheaper end deliberately: real precedent (Dr. GRPO's own global batch = 128 prompts/step; TinyZero's real low-budget reproduction = 256 prompts/step; Yue et al. 2504.13837's own VeRL controlled experiments = 256 prompts/step) all run considerably larger than 16 — meaning this choice trades a real, non-hypothetical increase in per-step gradient noise (and GRPO's known zero-reward-variance-group dead-signal risk, proportionally worse with fewer prompts) for roughly half the compute cost of the next step up (32 prompts/step, S=256). Documented as a deliberate budget-driven scope decision, not an oversight — revisit if Phase 1's training curve looks unusually noisy.
6. **Step budget**: target 500 steps (the plan's previous hard ceiling, now the target). Checkpoint every 100 steps, evaluate text-mode pass@1 at each checkpoint, stop early if plateaued across 2–3 consecutive checkpoints. Directly grounded in the foundational paper for this project (Yue et al. 2504.13837, Table 4, Omni-MATH-Train GRPO): their own step150/300/450 checkpoints show pass@1 still rising (26.1 → 33.6 → 42.5) and pass@256 still declining (66.3 → 65.3 → 64.3, their core sharpening signal) with no plateau through step 450 — 500 lands at/past their point of strongest observed effect. Given this project's smaller batch (S=128 vs. their controlled-experiment 256) increases per-step noise, erring toward more steps rather than fewer compensates somewhat. 300 steps remains a defensible fallback if budget forces a cut (their own step300 checkpoint already shows a real, smaller gain).
7. **Generation cap**: 1200 tokens (standardized across training and eval). Real measured natural completion length on this exact model, now covering all three conditions, not just text (2026-08-10, `scripts/check_natural_generation_length.py` then the larger `scripts/expanded_length_check_tde.py`, 280 real generations across T/D_transcribe/D_solve/E): T max 582, D_solve max 606, E max 615 - the highest value observed anywhere across every condition was 615 tokens, with a 2200-token generous cap never once hit. 1200 leaves comfortable margin (nearly 2x the observed max) while costing meaningfully less than the earlier 2000-token setting, given actual generation length - not the cap - drives compute cost (see the earlier real-throughput analysis: raising the cap only matters for the fraction of generations it would otherwise truncate). **Truncation-rate check mandatory**: measure % of generations hitting the cap without an extracted answer, separately per condition/model. If truncation is asymmetric (RL model truncating more than base - a length-drift symptom), raise the cap and re-run before trusting results. (Dr. GRPO's length-bias fix directly reduces this risk - see §7.) Real VRAM/throughput measurements taken at a 2000-token cap (`scripts/vram_smoke_test_2000cap.py`, `scripts/timing_calibration_2000cap.py`) remain representative at 1200: actual generated lengths in those real runs (438-562 tokens) never approached even 1200, so the cap value itself wasn't the limiting factor - only the finding it produced is: naive full-batch training OOMs on a single 40GB A100 (the LM head's vocab-size logits tensor, not KV cache, is the bottleneck) - confirmed a training micro-batch of 4 is the ceiling; training must use gradient accumulation to reach any larger effective batch size, not a single large batch.
8. **Manipulation check**: confirm vision encoder + projector weights are byte-identical before/after training. Report this explicitly.
9. **Statistical treatment**: compute Δ_text and Δ_pixel from the single seed's results, with a bootstrap/Bayesian CI on each Δ (§4 item 8) to quantify within-checkpoint sampling uncertainty. This CI reflects sampling noise only, not seed-to-seed variance (there is only one seed) — report it as such, not as a full uncertainty bound on "would another training run reproduce this."

**Compute estimate (1 seed, 500 steps, S=128 sequences/step, 1200-token cap, A100 40GB, from real Modal-measured throughput at a 2000-token cap - see the note in item 7 above on why these figures remain representative - `scripts/vram_smoke_test_2000cap.py` and `scripts/timing_calibration_2000cap.py`)**: training ≈ 11.1 hr, ~$41 (§9.2 table, S=128 row, 500 steps) + eval ≈ 7.2 hr, ~$26 (§9.3, 50 eval problems, locked) — **total ≈ 18.3 hr, ≈ $67**. This is the now-concrete Phase 1 figure per the locked group size (8) / prompts-per-step (16) / step count (500) / eval-problem-count (50) decisions above; still pending real-veRL calibration (not the raw-HF proxy this is based on) per the optimization discussion above the batch-shape item.

### Phase 2 — Second Model(s), Budget-Contingent

Only proceed here once Phase 1 produces a real, seed-validated Δ_text (i.e., the checkpoints aren't broken and there's a genuine signal to test). Two candidate second models, each with a distinct, motivated purpose — not just "more data points":

#### 2A — Qwen2.5-VL-7B (same-family scale check)

- **Purpose**: does the effect hold at a larger size within the *same* architecture family? Cleanest possible generalization check (no cross-family confound).
- **Risk**: memory is genuinely tight on a single 24GB GPU (L4) — real evidence found of instability (NaN gradients, zero rollout scores) attempting Qwen2.5-VL-7B GRPO on a 24GB-class GPU (A10). **Recommendation: run 7B on A100 40GB, not L4** — fits comfortably with plain LoRA there, no QLoRA needed.
- **Scope**: single-seed, matching the Phase 1 design (§3 Phase 1 item 4). Re-profile throughput at the 1200-token cap before committing compute — do not reuse 3B's numbers directly.

#### 2B — Qwen3-VL-2B-Instruct (different-generation, different-pedigree check)

- **Purpose**: this is not "a smaller model for cost reasons" — it's a **different research axis**: Qwen3-VL-2B's baseline reasoning was very likely acquired mostly via **Strong-to-Weak Distillation** from a larger already-RL'd teacher (Qwen3's smaller models use this route, unlike the larger ones which get the full SFT→RL→preference-RL pipeline). Testing whether the modality-shift effect holds on a model whose reasoning came from a *different training pedigree* is a genuinely motivated, thematically coherent comparison — not just "another model."
- **Must use**: the **Instruct** edition, never the **Thinking** edition (which already has heavy RL-for-reasoning baked in — would badly confound a before/after-our-RL comparison).
- **Mandatory pre-check, non-negotiable**: run the pass@1-vs-pass@16 headroom check (as in Phase 0, step 2) on Qwen3-VL-2B-Instruct specifically, **before any training compute is spent on it**. Distillation-acquired reasoning can already be "sharp" (high pass@1 close to its own pass@k ceiling), leaving little headroom for our GRPO step to move — this would show up as a small/noisy Δ_text that isn't a design flaw, just a ceiling effect on this specific model. Catch this cheaply first.
- **Tooling**: confirmed supported — EasyR1/veRL natively support Qwen3-VL.
- **Architecture caveat**: Qwen3-VL uses a different vision encoder (SigLIP2-Large) than Qwen2.5-VL — any cross-model comparison is confounded with architecture, same caveat as any cross-family check.
- **Outcome framing**: if 2B and 3B agree directionally — stronger, more general claim (robust across training pedigree, not just architecture). If they disagree — still a valid, arguably more interesting paper, reframed as "whether the effect holds depends on how the base model acquired its reasoning" — requires explicit discussion of the ceiling-effect / ownership-of-reasoning distinction, not presented as a clean failure.

**Compute estimate (2B, 1 seed, 300 steps, A100 40GB)**: stale — the figure previously here (≈8.2–13.6 hr for 3 seeds at a 500-token cap) predates the 1-seed decision and the later 1200-token cap. Re-profile with real timing calibration (as done for 3B in §9.2) before committing compute, rather than dividing the old number by 3.

---

## 4. Evaluation Pipeline (once inference is done)

1. **Calibration step, mandatory before trusting anything else**: run the untouched base model on GSM8K-V (or the relevant condition) and check the result is in the right ballpark of a known published number (e.g., Gemini-2.5-Pro scored 46.93% on GSM8K-V — not directly comparable to a 3B model, but a sanity check that the pipeline isn't producing absurd numbers like near-0% or near-100%). If this fails, debug the pipeline before trusting any RL-vs-base comparison built on top of it.
2. **Answer extraction**: regex-extract the model's final numeric answer, matching GSM8K's standard `#### [number]` convention. Normalize formatting (commas, decimals) before comparing.
3. **Correctness check**: exact numeric match against ground truth. (GSM8K/GSM8K-V's clean numeric answers avoid needing an LLM-judge or symbolic-equivalence tool — a deliberate, lower-risk choice.)
4. **Transcription fidelity** (Condition D only): compare the model's self-transcription against the ground-truth problem text (exact/normalized match) — gives a directly reportable "transcription accuracy" metric.
5. **Extraction spot-check**: manually review a sample of outputs from *each* condition (T/D/E) before trusting the regex at scale — a regex tuned on text-mode output isn't guaranteed to match image-mode phrasing habits.
6. **Sampling**: n ≈ 128 samples per problem (enough margin for pass@k up to k≈64). Temperature swept per model per k, not one fixed value for everyone (pass@k is documented to be temperature-sensitive).
7. **pass@k computation**: unbiased estimator (§2.1 formula), at minimum k=1 and k=64. **Do not brute-force k=512** — verified (§5) this isn't done anywhere for VLMs at this scale, and separately, GSM8K-level problems saturate pass@k early for capable models, so k=512 likely adds cost without adding signal. If a large-k number is wanted for comparison with the Yue-et-al.-style literature, use the extrapolation method (arXiv 2510.05197) instead of raw sampling.
8. **Uncertainty**: bootstrap or Bayesian confidence intervals on every reported Δ — never a bare point estimate.
9. **Supporting controls** (run on every condition/model/seed combination):
   - Perception-conditional bucketing (now largely built into the T/D/E design itself).
   - Difficulty-matched text control (find a text subset where base pass@1 ≈ base image-mode pass@1; check if Δ_RL is still large there).
   - Paraphrased-text OOD control (rules out "RL is just fragile to any distribution shift" vs. specifically modality).
   - Spurious-correctness spot check (~30–50 manually reviewed "correct" traces, checking for right-answer/wrong-reasoning).

---

## 5. What We Calculate, and What We Analyze

| # | What we calculate | What it's used to analyze |
|---|---|---|
| 1 | pass@1(π_base, T), pass@1(π_RL, T) | Δ_text — the reference gain, in the format RL trained on |
| 2 | pass@1(π_base, D/E), pass@1(π_RL, D/E) | Δ_pixel at each condition — the core novel measurement |
| 3 | pass@k(π_base, T/D/E), pass@k(π_RL, T/D/E), k up to ≈64 | Whether RL expands raw coverage, or only reshuffles precision (the sharpening signature), in each modality |
| 4 | Bootstrap/Bayesian CI on Δ_text, Δ_pixel, and their difference | Statistical validity — is the modality-shift effect real, not noise |
| 5 | Δ_text / Δ_pixel from the single seed, with bootstrap CI | Quantifies within-checkpoint sampling uncertainty (not seed-to-seed variance — single-seed design, see §10 item 8) |
| 6 | Transcription accuracy (Condition D) | Isolates pure perception error from reasoning error |
| 7 | Truncation rate per condition/model | Rules out the 1200-token cap as a hidden confound |
| 8 | Vision encoder + projector weight diff (before/after RL) | Confirms the manipulation ("text-only RL") was actually clean |
| 9 | Perception-conditional / difficulty-matched / paraphrase-OOD subsets | Rules out perception-tax, baseline-level, and general-fragility alternative explanations |
| 10 | Spot-checked reasoning traces (~30–50) | Rules out spurious correctness (right answer, wrong reasoning) inflating the numbers |
| 11 | (If 2nd model run) Same Δ_text/Δ_pixel comparison on 2B and/or 7B | External validity — does the effect generalize across architecture and/or training pedigree |

---

## 6. The Four Possible Outcomes — All Publishable

A good design lets you state, in advance, what every outcome means. All four below are real, evidenced-through-controls findings — none require "getting lucky."

| Outcome | Pattern | Meaning | Why it's publishable |
|---|---|---|---|
| **A — Collapse** | Δ_pixel ≈ 0, ≪ Δ_text; pass@k flat both modalities | RL's gain is tied to the text format it trained on | Cautionary, practical finding — RL-trained VLM benchmarks may not transfer to real deployment (photos of homework, whiteboards) |
| **B — Survival** | Δ_pixel ≈ Δ_text, both clearly >0 | RL taught something that survives regardless of input format | The most surprising, positive-for-RL result — evidence against sharpening on an untested axis |
| **C — Partial** | 0 < Δ_pixel < Δ_text, both significant and distinct | Some but not all of the gain transfers | Arguably the most realistic/useful outcome — gives a real *quantity* ("X% survives"), not a forced binary |
| **D — Negative transfer** | Δ_pixel < 0 | RL actively hurts image-mode performance | Points to a distinct mechanism (narrow-distribution adapter interference, not pure sharpening) — connects to a different, also-interesting literature |

**The only non-automatically-valuable outcome**: pure noise (CIs too wide to distinguish any of the above at the tested sample size). This is a real, sharper risk under the single-seed design (§3 Phase 1 item 4, §10 item 8) than it would be with multiple seeds — Phase 0's smoke test exists to catch obviously-broken setups cheaply before the full investment, but cannot substitute for the seed-variance check a multi-seed design would have provided.

---

## 7. Algorithm Choice: Dr. GRPO (not vanilla GRPO, not CISPO)

**Use Dr. GRPO.** It fixes two documented biases in vanilla GRPO:
1. **Length bias**: vanilla GRPO under-penalizes long wrong answers and over-rewards short right ones (a normalization artifact), causing generations to drift longer over training for reasons unrelated to actually solving problems. Dr. GRPO normalizes by a constant instead, removing this.
2. **Difficulty bias**: vanilla GRPO over-weights very easy/very hard questions (low reward variance) via std-normalization. Dr. GRPO removes this too.

**Why this matters for us specifically, not just in general**: fixing the length bias directly reduces the risk of asymmetric truncation between base and RL model (§3, Phase 1 step 6) — a confound we'd otherwise have to catch and correct after the fact. It also directly reduces compute, since generation length is the dominant driver of training wall-clock time, and length-inflation was the single biggest source of our own earlier compute-estimate errors.

**Tooling**: confirmed natively supported in veRL (which EasyR1 is built on) — a config choice, not new engineering.

**Framing bonus**: Dr. GRPO comes from the same critical-perspective literature as the sharpening hypothesis itself ("Understanding R1-Zero-Like Training: A Critical Perspective," arXiv 2503.20783) — citing it strengthens, not complicates, our positioning.

**Not CISPO**: CISPO (from MiniMax-M1) is designed for large-scale, long-sequence, MoE training instability — not our regime (small dense model, capped short sequences, resource-constrained). The literature explicitly recommends GRPO-family methods for resource-constrained settings and CISPO for large/long/MoE settings. Switching would also detach us from the step-count precedent we've gathered, which is specifically about GRPO-family training.

---

## 8. Novelty & Related Work — Fully Verified

Every claim below was checked against the primary source, not just a search summary (two mistakes were caught and corrected this way — see the last two rows).

| Work | What it actually does | Why it's not a match |
|---|---|---|
| GSM8K-V | Paired text/image GSM8K, measures the raw modality gap (e.g. Gemini-2.5-Pro: 95.2% text vs. 46.9% image) | Static accuracy comparison, no RL involved at all — this is *infrastructure* we use, not competition |
| VisTIRA (2601.14440) | Closes the text/image math gap via external OCR/tool integration | Different mechanism (bolt-on tool, not testing the model's own learned reasoning) |
| Vision-R1, R1-Onevision | Address "modality gap" via cold-start CoT data + RLVR | Data-centric bridging, not a text-vs-image counterfactual on one checkpoint |
| Perception-R1, VTPerception-R1, ViGoRL, VL-PRM | Visual-grounding/perception process rewards for VLM reasoning | Different mechanism (reward design), not a sharpening-diagnostic comparison |
| CrossMath / "Rigorous Study of the Modality Gap" (2604.16256) | Matched text/image/combined benchmark (grid-inference puzzles), evaluates existing VLMs | Confirmed: pure benchmark/evaluation paper, **no RL training component at all**, different task domain |
| "When Does RL Help Medical VLMs?" (2603.01301) | Computes pass@k (K∈{1,2,4,8,16}) for base vs. RL, explicitly finds "RL mainly acts as sharpening" | **Verified via direct fetch**: their "modality" means *medical imaging scan type* (X-ray/CT/MRI/OCT), not text-vs-image. Confirmed zero text-only comparisons exist in the paper. Not a match — a coincidentally similar method applied to a genuinely different axis |
| "When RLVR Shrinks the Reasoning Boundary" (2607.20543) | Pass@k-inversion diagnostic for RLVR | **Correction of an earlier search-summary error**: verified via direct fetch this paper is purely text-only math (Qwen2.5-7B); the one VLM mention is an explicit "future work" sentence in the introduction, not an actual experiment |
| **MIRROR (2607.21552)** | Cross-modal teacher-student distillation training method; has a "text-dominant GRPO" baseline (vanilla, no cross-modal KL) evaluated on both text and image test sets | **Verified via direct fetch of Table 2**, exact numbers below — closest neighbor found, but confirmed not a match on the actual research question (see below) |

### MIRROR — the full, verified comparison

Table 2 (Qwen3-VL-4B, vanilla single-view GRPO baseline vs. untrained base):

| | Image pass@1 | Image pass@16 | Text pass@1 | Text pass@16 |
|---|---|---|---|---|
| Base model | 12.30 | 42.57 | 32.91 | 80.22 |
| Text-dominant GRPO | 17.32 | 43.29 | 39.92 | 81.06 |

Computed from their numbers: **Δ_text(pass@1) = 7.01, Δ_pixel(pass@1) = 5.02** (≈72% of the gain survives — "partial"); **Δ_text(pass@16) = 0.84, Δ_pixel(pass@16) = 0.72** (both essentially flat — consistent with the sharpening signature in both modalities).

**Why this is motivating evidence, not a scoop**:
1. **Nobody computed this** — it required subtracting numbers from a baseline table in a paper about a different method entirely.
2. **No sharpening framing at all** — confirmed, no citation of Yue et al., no pass@1-vs-pass@k gap analysis, pass@16 used purely as a headline accuracy number.
3. **Curated, not natural, dataset** — confirmed directly: ODA-Val is filtered to keep only "modality-dependent" problems (solvable in one view, not the other) — a biased sample, unlike our natural, unfiltered GSM8K-V.
4. **Single run, no seeds reported for this baseline.**
5. **Different model (Qwen3-VL-4B) and domain (geometry + LLM-generated TikZ diagrams, not arithmetic word problems).**
6. **Confirmed explicitly in the paper's own limitations**: "the paper does NOT decompose whether improvements are modality-specific... or genuinely cross-modal... measures aggregate consistency but not directional transfer attribution" — that directional attribution is exactly Δ_text-vs-Δ_pixel, i.e., exactly our contribution, stated by them as something they don't do.

**Action item**: cite this MIRROR computation explicitly in Related Work / motivation — it strengthens the paper by showing real engagement with the closest prior evidence, and shows the effect is plausible before we've spent any compute confirming it rigorously.

### The "raw number vs. finding" principle (why this is still valid research)

A number sitting unanalyzed in someone else's baseline table is not the same as a tested finding. What we add: a *natural* dataset (not curated/biased), *multiple seeds* (not one run), the *sharpening-hypothesis framing* (explicitly connecting to Yue et al. and the active debate), and the *specific confound controls* (perception-conditional, difficulty-matched, truncation-rate, decomposed T/D/E design) that let us actually trust a number rather than just observe it. This is the same pattern Yue et al.'s own paper followed — formalizing informal community suspicion into a rigorously controlled, citable result.

**Standing caveat**: this is a fast-moving, preprint-heavy field. Re-run this novelty check close to submission (arXiv listings, Semantic Scholar citation graph off the papers above, OpenReview).

---

## 9. Compute Analysis

### 9.1 Hardware profile

| | L4 | A100 40GB |
|---|---|---|
| VRAM | 24 GB | 40 GB |
| Memory bandwidth | ~300 GB/s | ~1,555 GB/s |
| BF16 compute | ~121 TFLOPS | ~312 TFLOPS |
| Typical cloud price | ~$0.20–$0.80/hr | ~$1.50–$3.00/hr |

### 9.2 Training — 1 seed, A100 40GB, 1200-token cap

Real measured throughput, not a blind estimate (2026-08-10, Qwen2.5-VL-3B, `scripts/timing_calibration_2000cap.py` + `scripts/vram_smoke_test_2000cap.py`, both run at a 2000-token cap but representative at 1200 too - see item 6 in §3 Phase 1): rollout = 864.8 tok/s aggregate at 64 concurrent sequences; training = 0.48s per micro-batch-of-4 step (4 is the confirmed VRAM ceiling at this response length on a single 40GB A100 — larger micro-batches OOM on the LM head's vocab-size logits tensor, not KV cache). Per-step time = (S/64)×32.42s [rollout] + (S/4)×0.48s [training], where S = prompts-per-step × group_size — not yet fixed (§12), shown below for two illustrative S values:

| S (sequences/step) | 300 steps | 400 steps | 500 steps |
|---|---|---|---|
| 128 | 6.7 hr, ~$25 | 8.9 hr, ~$33 | 11.1 hr, ~$41 |
| 256 | 13.4 hr, ~$50 | 17.8 hr, ~$66 | 22.3 hr, ~$83 |

2B/7B rows removed: Phase 2 is budget-contingent and decided only after Phase 1 (§3.4). The old 500-token-cap estimates that used to be here are stale now that 3B has been re-profiled - re-run the same real timing-calibration approach for whichever model is actually pursued in Phase 2, rather than reuse a blind estimate.

### 9.3 Evaluation — 1 seed, both checkpoints (base + RL), all 3 conditions T/D/E, n≈128, 1200-token cap, A100 40GB

Real measured rollout throughput (864.8 tok/s aggregate, §9.2) applied to: eval_problems × n(128) × call_units(T=1, D=2, E=1 → 4 total) × avg_completion_tokens(438, real measured natural length, §9.2).

**eval_problems = 50, locked** (2026-08-12) — a deliberate, budget-driven compromise: real precedent (Yue et al. 2504.13837's own eval sets — 821 in-domain, 460 MathVista, 114 MathVision, the latter two being real visual/image-math benchmarks directly relevant to our D/E conditions) all run considerably larger and would cost proportionally more (114 alone ≈ $59 eval / ~$100 total, 460/821 far beyond scope); 50 was chosen as the largest count the available compute budget supports, trading away some of that statistical power (bootstrap CI on Δ_text/Δ_pixel will be wider, resampling from fewer problems) for affordability. Documented as a named scope limitation, same treatment as the single-seed decision (§3 Phase 1 item 4) — not presented as a "sufficient" sample size, just the budget-feasible one.

| Model | Estimated time (50 eval problems, both checkpoints) | Estimated cost |
|---|---|---|
| 3B (Qwen2.5-VL-3B) | ~7.2 hr | ~$26 |

2B/7B rows removed for the same reason as §9.2.

### 9.4 Recommended sequencing

1. Phase 0 (3B smoke test): trivial cost, <1 hour of actual compute.
2. Phase 1 (3B, 1 seed, 500 steps, S=128, 50 eval problems, 1200-token cap): **~18.3 hr total, ~$67** (real-throughput-based, §9.2+§9.3) — the core, load-bearing result.
3. Phase 2A (7B, if budget allows): run on **A100 40GB, not L4** (confirmed real instability risk on 24GB-class GPUs at this model size). Re-profile throughput at the 1200-token cap before committing compute — do not reuse the old 500-token-cap estimate.
4. Phase 2B (2B, if budget allows): run the mandatory headroom pre-check first (near-zero cost), then proceed only if it passes. Re-profile throughput at the 1200-token cap before committing compute.

---

## 10. Critical Validity Checklist — Conditions for This to Be a Sound Paper

1. **Report the weight-identity manipulation check** (vision pathway untouched) — for every seed.
2. **Include perception-conditional bucketing** (via Condition D) and the **difficulty-matched text control** — rules out "images are just harder" and "the baseline was just lower" as alternative explanations.
3. **Add the paraphrased-text OOD control** — separates "modality-specific fragility" from "fragile to any distribution shift."
4. **Report confidence intervals**, never bare point estimates — pass@k is documented to be unstable at small sample sizes.
5. **Frame the contribution honestly**: *"testing whether the pass@1/pass@k signature generalizes across modality"*, not *"proving sharpening is/isn't true"* — the underlying methodology (pass@k as a capability-boundary proxy) is itself actively contested in the literature (arXiv 2511.16231, 2607.20543), so claim only what the design can actually support.
6. **Name the adapter-transfer-failure alternative explicitly in Limitations** — a LoRA update trained only on text-shaped inputs could fail to transfer to image-conditioned inputs for reasons unrelated to sharpening (narrow-distribution interference). A collapse result is *consistent with* representation-bound sharpening, not proof of it, unless this alternative is explicitly discussed.
7. **Scope every claim to the tested model(s)** — do not generalize beyond what was actually run; this mechanism is plausibly architecture-dependent by its nature (vision-language calibration quality could change the outcome independent of the "true" sharpening question).
8. **Single-seed result (3B)** — a deliberate scope decision (§3 Phase 1 item 4), not the original multi-seed design. Report this explicitly as a limitation: the reported Δ_text/Δ_pixel reflect one training run's sampling-level uncertainty (via bootstrap CI, item 4 above) but cannot characterize seed-to-seed variance, which is documented to shift pass@1 by several percentage points in comparable LLM-reasoning RL settings. State this plainly in the paper's limitations section rather than implying the CI covers it.

---

## 11. Datasets

- **Primary**: GSM8K-V — paired 1:1 with GSM8K text problems, public, numeric `#### [number]` ground truth, exact-match verifiable. Fix one consistent rendering configuration (font, DPI, layout) for the main result — document it precisely (rendering choices can swing accuracy by tens of points); treat other renderings as a sensitivity-appendix only.
- **Stretch goal**: a rendered MATH-500 subset, if the GSM8K-V pilot succeeds and more headroom/difficulty range is wanted.

---

## 12. Open Questions / Next Decisions for the Team

- Final LoRA hyperparameters (rank, target modules, learning rate) and Dr. GRPO settings (KL coefficient) — decide before Phase 0. (Group size is now locked at 8 — see §3 Phase 1 item 5.)
- ~~Prompts-per-training-step~~ — **locked**: 16 (group size 8, S=128 sequences/step, 500-step target — §3 Phase 1 items 5-6, real precedent and rationale there).
- ~~eval-problem-count~~ — **locked**: 50, a deliberate budget-driven compromise below real precedent (§9.3 for full rationale and cost table) — expect a wider bootstrap CI on Δ_text/Δ_pixel than the paper's own 821/460/114 would give.
- ppo_epochs — still not fixed (real Dr. GRPO precedent: 1, already set explicitly in `src/training/phase1_config.py`; revisit only if there's a specific reason to deviate).
- Workshop target and deadline — determines how much of §10's checklist is feasible in the available time (priority order if constrained: manipulation check → perception-conditional/D-condition → confidence intervals → paraphrase OOD control → spurious-correctness spot check).
- Whether to pursue Phase 2A (7B), Phase 2B (2B), both, or neither — decide **after** Phase 1 results are in, not before.
- Re-run the novelty check (§8) close to submission.

---

*This is a living document — update it as Phase 0/1/2 results come in, rather than treating it as fixed.*
