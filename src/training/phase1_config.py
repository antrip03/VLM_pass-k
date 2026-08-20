"""
Real Phase 1 veRL Dr. GRPO training config (PLAN.md Section 3, Phase 1) -
the production run, not the tiny plumbing check in dry_run_config.py.

Batch shape locked 2026-08-11 (PLAN.md Section 3 Phase 1 item 5): group
size 8, prompts-per-step 16 -> 128 total sequences/rollout (S=128).
ppo_mini_batch_size is set equal to the full rollout (128), not left at
some smaller default, so that ONE gradient update per rollout is computed
by accumulating over all 128 sequences (via use_dynamic_bsz + fused
kernels), rather than several smaller, noisier mini-batch updates -
confirmed via veRL's own docs (verl.readthedocs.io/en/latest/examples/config.html,
fetched verbatim 2026-08-11): "ppo_mini_batch_size is a global num across
all workers/gpus" (resolves real, unanswered confusion visible in
volcengine/verl issues #1524, #2266, #3288 on this exact point - none of
those threads got a maintainer answer, so this project verified against
the docs directly instead of trusting community guesses). The actual
per-forward-pass memory ceiling remains governed independently by
ppo_max_token_len_per_gpu + use_dynamic_bsz (already true in dry_run_config.py),
so raising ppo_mini_batch_size to 128 does not by itself increase VRAM
risk - confirmed via the real ray_trainer.py divisibility assertion
(real_train_batch_size = train_batch_size * rollout.n must divide evenly
by n_gpus; 16*8=128, n_gpus=1 here, trivially satisfied) and via GRPO's
advantage computation happening once over the full rollout BEFORE any
mini/micro-batch splitting (so group-relative advantage correctness is
unaffected by this choice either way).

ppo_epochs=1 set explicitly (not left to an unverified default) -
matches real Dr. GRPO precedent (sail-sg/understand-r1-zero's
train_zero_math.py: --num_ppo_epochs 1).

lr=1e-6, use_kl_loss=False, lora_rank=32/lora_alpha=32 locked 2026-08-12
(PLAN.md Section 12): lr/KL switched from veRL's generic LoRA+vision
example defaults (3e-6, kl_loss_coef=0.01) to Dr. GRPO's own real recipe
(train_zero_math.py: learning_rate=1e-6, beta/KL=0.0) - independently
better-justified here too, not just "the paper's number": our single-GPU
per-step batch (S=128) is a noisier gradient estimate than Dr. GRPO's real
1024-sequence global batch (8 GPUs), arguing for the more conservative LR;
and nonzero KL pulls the trained policy back toward the base model,
directly damping Delta_text - the exact quantity this project measures -
rather than guarding against a real evidenced risk (DeepSeek-R1-Zero's
actual documented pure-RL issues were repetition/readability/language-
mixing, not correctness or instruction-following collapse, with
"little to no evidence of incoherence... on domains similar to math and
coding" i.e. within-domain, which is exactly this project's GSM8K case).
LoRA rank switched from 64 (also a veRL-example default, unverified for
RLVR) to 32 per real RLVR+LoRA ablation evidence (arXiv 2512.23165: ranks
16 and 32 land in the same 42-44% accuracy range - performance plateaus
well before 32, r=1 clearly underperforms at 40.5% - and a separate
finding that LoRA+RL accuracy stagnates or declines above the 32-64
range, attributed to gradient entanglement from the low-rank bottleneck).
alpha=32 kept equal to rank (1:1 scaling, the standard LoRA convention) -
no RLVR-specific evidence found for a different ratio, so no reason to
keep the previous 64/32=0.5 damped ratio that was itself just inherited,
not chosen.

Every FSDP/dtype/vision-exclusion bug fix from dry_run_config.py carries
over unchanged - those are correctness fixes for this exact model on this
exact veRL build, not dry-run-specific.

NOT yet done: real per-checkpoint evaluation is deliberately NOT wired
through veRL's own internal validation loop (trainer.test_freq=-1, same
as the dry run) - this project's own sampling/metrics harness (Step 6/7:
src/inference/run_sampling.py, src/metrics/bootstrap_ci.py) is more
capable (T/D/E conditions, pass@k, bootstrap CI) than veRL's built-in val
loop, so checkpoints are evaluated offline with that harness instead,
using trainer.save_freq to control how often a checkpoint exists to
evaluate. wandb logging enabled 2026-08-12 (see the trainer.logger line
below for the real mechanism and why no key ever appears in this file).
"""

from __future__ import annotations


def build_args(
    *,
    model_path: str,
    train_file: str,
    val_file: str,
    reward_fn_path: str,
    checkpoint_dir: str,
    prompts_per_step: int = 16,
    group_size: int = 8,
    max_prompt_length: int = 512,
    max_response_length: int = 1200,
    ppo_max_token_len_per_gpu: int = 8192,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    rollout_gpu_mem_util: float = 0.45,
    total_training_steps: int = 500,
    checkpoint_every: int = 100,
    seed: int = 0,
    val_before_train: bool = True,
) -> list[str]:
    """
    Returns the full Hydra CLI arg list for `python3 -m verl.trainer.main_ppo`.

    seed=0 (PLAN.md's "1 seed, seed 0" decision - §3 Phase 1 item 4) is
    wired through data.seed, the one seed-related key confirmed in veRL's
    real docs (verl.readthedocs.io/en/latest/examples/config.html) that's
    directly relevant here - it pins GSM8K's shuffling order, which was
    previously left at its default (null/unseeded), meaning two runs of
    this same function would have silently produced different data orders.
    Honest scope limit: this does NOT claim full bit-for-bit
    reproducibility of the whole run - LoRA weight init and vLLM rollout
    sampling have their own randomness sources, and no single documented
    "seed everything" veRL key was found covering those too. "1 seed" in
    this project's design was always about study scope (not characterizing
    seed-to-seed variance across multiple runs), not a stronger
    bit-reproducibility promise - this just makes the one real, confirmed
    lever actually pinned instead of silently left unset.

    S (sequences/rollout) = prompts_per_step * group_size = 16 * 8 = 128,
    matching PLAN.md Section 9.2's S=128 row (11.1 hr / ~$41 at 500 steps,
    raw-HF-proxy-based - real veRL calibration still pending, see the
    optimization discussion this config follows from).
    """
    ppo_mini_batch_size = prompts_per_step * group_size  # global, = full rollout (S) - see module docstring
    return [
        # ---- algorithm / data ----
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=False",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={prompts_per_step}",  # PROMPTS, not total sequences - confirmed via real verl source (ray_trainer.py) and docs
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"data.seed={seed}",  # pins data-shuffling order - see module docstring on what this does/doesn't cover
        # ---- model / LoRA / vision exclusion ----
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.model.lora_rank={lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear",
        # lm_head exclusion TRIED AND DISPROVEN 2026-08-19: excluding the
        # LM head from LoRA did NOT stop the FSDP tied-parameter assertion
        # (same error, same post-forward reshard site). The tie is a
        # property of the BASE model (tie_word_embeddings=true), not of
        # LoRA wrapping - so excluding lm_head from the adapter was never
        # going to change FSDP's shared-parameter bookkeeping. Reverted to
        # the original visual-only exclusion; the real fix is strategy=fsdp2
        # below. Kept quoted-regex-free (no parens/pipes) since Hydra's
        # override grammar rejects bare parentheses.
        "actor_rollout_ref.model.exclude_modules=.*visual.*",
        "actor_rollout_ref.model.use_fused_kernels=True",
        # ---- actor / Dr. GRPO fixes ----
        # lr=1e-6 and use_kl_loss=False (2026-08-12): switched from veRL's
        # generic LoRA+vision example defaults (3e-6, kl_loss_coef=0.01) to
        # match Dr. GRPO's own real, evidence-backed recipe (train_zero_math.py:
        # learning_rate=1e-6, beta/KL=0.0) - not just "the paper's number" but
        # independently better-justified for this project specifically: our
        # single-GPU per-step batch (S=128) is a noisier gradient estimate than
        # Dr. GRPO's real 1024-sequence global batch (8 GPUs), which argues for
        # the more conservative LR, not the higher one; and a nonzero KL pulls
        # the trained policy back toward the base model, directly damping
        # Delta_text - the exact quantity this project measures - rather than
        # protecting against a real, evidenced risk (DeepSeek-R1-Zero's actual
        # documented pure-RL issues were repetition/readability/language-mixing,
        # not correctness or instruction-following collapse, and specifically
        # showed "little to no evidence of incoherence... on domains similar to
        # math and coding" - i.e. within-domain, which is exactly our GSM8K
        # case). Residual RLVR risk this does NOT address (right-answer/
        # spurious-reasoning shortcuts) is a separate, already-covered item -
        # PLAN.md Section 10's spurious-correctness spot check, not a KL
        # question.
        # strategy=fsdp2 (2026-08-19) - the REAL fix for the tied-parameter
        # crash that neither param_offload=False nor excluding lm_head from
        # LoRA resolved. Read from PyTorch's own source: the assertion lives
        # in FSDP1's _flat_param.py, in the loop over _shared_param_infos,
        # and requires the PRIMARY of a tied pair to still be an nn.Parameter
        # when this handle reshards with as_params=True. During
        # forward/backward FSDP1 registers params as bare Tensors
        # (as_params=False -> _setattr_tensor, "hide originals from
        # named_parameters()"), so when the tied pair
        # (embed_tokens.weight <-> lm_head.weight, tie_word_embeddings=true
        # on Qwen2.5-VL) is split across two different FSDP units, one unit
        # reshards while the other still holds a plain Tensor -> assertion.
        # That is a property of the base model's weight tying and FSDP1's
        # flat-parameter bookkeeping, which is why neither earlier fix
        # touched it. FSDP2 shards per-parameter via DTensor and has no
        # flat_param/_shared_param_infos machinery at all, so this code path
        # does not exist there. LoRA + vLLM rollout is documented as
        # supported under strategy=fsdp2 (verl.readthedocs.io ppo_lora).
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.ref.strategy=fsdp2",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_epochs=1",  # explicit, matches real Dr. GRPO precedent - not left to an unverified default
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
        # param_offload=False (was True) - real crash found on a live Modal
        # run 2026-08-19: "AssertionError: as_params=True
        # type(prim_param)=<class 'torch.Tensor'>" in PyTorch FSDP internals
        # (_flat_param.py, _use_unsharded_views), reached via
        # offload_fsdp_model_to_cpu, i.e. exactly this offload path -
        # crashed on the very first actor update step (before step 25,
        # before any checkpoint - initial validation had already passed
        # cleanly at 63.7% GSM8K, confirming the data/reward pipeline
        # itself was fine). Root cause, confirmed via direct inspection of
        # the actual PyTorch source (not the two GitHub issues first cited
        # for this, which turned out NOT to actually support the claim on
        # direct check - #4418 is an unrelated vLLM LoRA rollout assertion,
        # #2655 is an unrelated missing-offload-for-ref-policy feature
        # request): this assertion is specifically about SHARED/TIED
        # parameters (_shared_param_infos, "primary owner" tracking).
        # Qwen2.5-VL-3B-Instruct ties its word embeddings (confirmed via
        # its real config.json: tie_word_embeddings=true) - since
        # target_modules=all-linear wraps the LM head in a LoRA adapter,
        # the tied embedding/LM-head pair very likely ends up with
        # inconsistent nn.Parameter/Tensor typing specifically when
        # param_offload's post-step reconstruction tries to rebuild that
        # shared parameter's "primary owner". Disabling this offload
        # avoids the code path entirely regardless of the exact mechanism.
        # Real, adjacent corroborating evidence found independently:
        # pytorch/pytorch#91165 (open, high-priority) documents FSDP +
        # CPU offload + frozen-parameter models as a genuinely fragile,
        # actively broken combination generally, consistent with this.
        # rollout_gpu_mem_util lowered 0.6->0.45 (default above) to free
        # VRAM headroom to compensate for the actor's params no longer
        # being offloaded - untested at the time of this fix, verify with
        # a cheap short run before trusting a long one.
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=True",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
        f"actor_rollout_ref.actor.loss_scale_factor={max_response_length}",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]",
        # ---- rollout (vLLM) ----
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_mem_util}",
        f"actor_rollout_ref.rollout.max_model_len={max_prompt_length + max_response_length + 100}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.n={group_size}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        # disable_cascade_attn=True (2026-08-14): real, evidenced fix for a
        # documented vLLM training-inference mismatch. Cascade attention is
        # vLLM's shared-prefix batch-decoding optimization - exactly our
        # situation, since group_size=8 means every rollout batch has 8
        # sequences sharing the same prompt. A real report (Qwen3-14B-Base,
        # on-policy GRPO, A100s) found this specific vLLM kernel path
        # produces log-probs that don't exactly match the training engine's
        # for the same tokens, and that disabling it dropped their
        # mismatch metric (vllm-kl) from the 5e-2 to 1e-1 range down to
        # ~1e-3, with rewards also improving - not just numerically
        # cleaner, a real training-quality effect. "+" prefix required
        # (this key isn't in veRL's base config schema, confirmed via the
        # real recommended fix: "+actor_rollout_ref.rollout.engine_kwargs.
        # vllm.disable_cascade_attn=True" for exactly this symptom,
        # rollout_probs_diff_mean being too high). Real cost: gives up a
        # secondary vLLM decode optimization, so some rollout slowdown is
        # plausible - no precise number found, but this targets a root
        # cause found to meaningfully affect actual training quality, not
        # just a cosmetic fix, so it's worth the unquantified cost.
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True",
        # calculate_log_probs=True (2026-08-14): required for veRL's own
        # rollout_corr/* metrics (kl, k3_kl, chi2_token, chi2_seq,
        # log_ppl_abs_diff, ppl_ratio) to be computed at all - confirmed
        # via verl.readthedocs.io/en/latest/algo/rollout_corr.html. This
        # gives real visibility into residual training-inference mismatch
        # in wandb (trainer.logger already includes wandb) without any
        # extra plumbing - real, documented cost is small (~1% memory,
        # 1-3% compute). Deliberately NOT also turning on active
        # importance-sampling correction (algorithm.rollout_correction.
        # rollout_is) yet - that reweights the loss itself, a more
        # consequential, unverified-for-this-project mechanism change:
        # if disable_cascade_attn already gets the real mismatch down to
        # the reference report's ~1e-3 level, active correction may not be
        # needed at all. Watch rollout_corr/kl and rollout_corr/
        # log_ppl_abs_diff during the calibration run and decide from real
        # data, not preemptively.
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        # ---- reference policy ----
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.entropy_from_logits_with_chunking=True",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16",
        "actor_rollout_ref.ref.fsdp_config.use_orig_params=True",
        # ---- trainer ----
        "trainer.balance_batch=True",
        # wandb enabled 2026-08-12 (real syntax confirmed against
        # docs.wandb.ai/weave/guides/integrations/verl): reads WANDB_API_KEY
        # from the environment at runtime - the key itself is never passed
        # as a CLI arg or written into any file, only exported in the VM's
        # own shell / passed through by scripts/gcp/run_training.sh's
        # `docker run -e WANDB_API_KEY`. If WANDB_API_KEY isn't set when
        # this runs, veRL/wandb will error asking you to log in - that's
        # expected until a real (rotated) key is exported on the VM.
        "trainer.logger=[console,wandb]",
        "trainer.project_name=grpo-vlm-modality-shift",
        "trainer.experiment_name=qwen2_5_vl_3b_phase1",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.save_freq={checkpoint_every}",
        "trainer.test_freq=-1",  # our own T/D/E harness evaluates checkpoints offline instead - see module docstring
        # val_before_train: veRL runs a full validation pass over all 1319
        # GSM8K test examples BEFORE training step 1 (test_freq=-1 only
        # disables the periodic in-loop validation, not this one). Measured
        # at 7.4 min on a real run, and longer since gpu_memory_utilization
        # dropped to 0.45. Genuinely useful as a baseline for the real run,
        # pure dead time for a short smoke test whose only question is
        # whether the first training step crashes - hence configurable.
        f"trainer.val_before_train={val_before_train}",
        # total_epochs=1, with total_training_steps=467 as the real target
        # (decided 2026-08-19). GSM8K's train split is 7473 examples and
        # data.train_batch_size=16 prompts/step, so ONE epoch is exactly
        # 7473/16 = 467.06 -> 467 steps. The original config asked for 500
        # steps with total_epochs=1, which is unreachable: two runs stopped
        # dead at 467, and veRL's end-of-epoch DataLoader teardown surfaced
        # as a misleading "RuntimeError: DataLoader worker ... killed by
        # signal: Killed" that reads like an OOM but is not.
        # Raising to 2 epochs would reach 500, but the last 33 steps would
        # be a SECOND pass over the first ~528 examples - every prompt seen
        # once except those, seen twice. 467 is the cleaner experiment:
        # exactly one pass, every prompt seen exactly once, no repetition.
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={total_training_steps}",
        f"trainer.default_local_dir={checkpoint_dir}",
        # resume_mode=auto (was disable) 2026-08-12: checkpoints were
        # already being saved every checkpoint_every steps, but nothing
        # was wired to actually USE them on a restart - a crash meant
        # starting over from step 0, not resuming. auto makes veRL look
        # for and resume from the latest checkpoint under
        # trainer.default_local_dir automatically. NOT yet verified with a
        # real kill-and-restart test - do that cheaply (a few steps, kill
        # the process, restart, confirm it resumes from the right step)
        # before trusting this for the real 500-step run, same standard
        # applied to every other untested mechanism in this project.
        "trainer.resume_mode=auto",
        "trainer.critic_warmup=0",
        # ---- custom reward ----
        f"custom_reward_function.path={reward_fn_path}",
        "custom_reward_function.name=compute_score",
    ]
