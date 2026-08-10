"""
Tiny, cheap end-to-end veRL Dr. GRPO dry run (PLAN.md Section 3, Phase 0
item 4: "a tiny GRPO/Dr. GRPO training pass ... just to confirm the
training loop runs, rewards compute correctly, and loss moves in a sane
direction"). NOT the real Phase 1 run - deliberately tiny batch sizes and
step count, single A100 GPU.

Originally written as a shell script (run_dry_run.sh); converted to a
plain Python arg-list builder after a real bug: Modal's
add_local_python_source("src", "configs") only mounts .py files into the
container (it's specifically for importable Python packages), so the .sh
file silently never made it in ("bash: .../run_dry_run.sh: No such file
or directory" on a live run, 2026-08-10). This sidesteps that whole class
of problem rather than finding a different way to ship a non-Python file.

Structure adapted from TWO real, independently-verified sources (not
guessed), both fetched verbatim 2026-08-10:
  1. examples/tuning/lora/run_qwen2_5_vl_7b_fsdp.sh (real, official veRL
     LoRA+vision example) - LoRA/vision-exclusion settings, actor/
     rollout/ref structure.
  2. modal.com/docs/examples/grpo_verl (real, complete, working
     Modal+veRL+GRPO example, including a literal dry-run config with
     total_training_steps=1) - overall data/trainer structure and the
     pattern of using a tiny step count for a plumbing check before any
     real run.

Differences from source (1), and why:
  - n_gpus_per_node=1, rollout tensor_model_parallel_size=1 (single A100,
    not the source's 8 GPUs)
  - model path = Qwen2.5-VL-3B-Instruct (this project's model, not 7B)
  - max_response_length=1200 (this project's real-data-derived cap,
    PLAN.md Section 3 Phase 1 item 6 - not the source's 2048)
  - train_batch_size / ppo_mini_batch_size / rollout_n set small and
    specifically for this dry run, not the eventual real Phase 1 values
    (those depend on Section 9's still-open S parameter - PLAN.md
    Section 12)
  - use_fused_kernels=True and entropy_from_logits_with_chunking=True
    added - NOT present in source (1) at all. This project's own real
    VRAM testing (scripts/vram_smoke_test_2000cap.py) found a raw-HF
    proxy hits an OOM wall at large batch sizes from the LM head's
    vocab-size logits tensor; both flags are veRL's real, documented
    answers to exactly that (use_fused_kernels: "fused cross entropy
    kernel to drastically reduce peak memory... avoids materializing the
    full logit tensor" - the more directly relevant of the two;
    entropy_from_logits_with_chunking: a narrower, entropy-specific
    chunking mechanism, kept as a secondary measure since it doesn't
    conflict). NEITHER has been verified yet against our exact
    bottleneck - that's what this dry run is for.
  - loss_agg_mode=seq-mean-token-sum-norm + loss_scale_factor=1200 +
    norm_adv_by_std_in_grpo=False added - Dr. GRPO's two specific bias
    fixes (PLAN.md Section 7), absent from source (1) (plain GRPO, not
    Dr. GRPO). norm_adv_by_std_in_grpo's exact config path
    (algorithm.norm_adv_by_std_in_grpo, inferred from the config
    hierarchy - adv_estimator and use_kl_in_reward both live under
    algorithm.*) is the one flag here NOT directly confirmed against a
    real example script - Hydra raises a clear "key not found" error if
    a config path is wrong, which is exactly what this dry run checks.
  - custom_reward_function pointed at this project's own
    src/training/reward_fn.py (uses the #### + \\boxed{} fallback chain
    already validated in src/metrics/answer_extraction.py), not veRL's
    built-in gsm8k reward (confirmed via real source inspection to
    default to "#### only" matching - the exact gap this project already
    found and fixed once).
  - trainer.logger=['console'] only (no wandb) - avoids needing a wandb
    secret for a plumbing check; real Phase 1 runs should add it back.
"""

from __future__ import annotations


def build_args(
    *,
    model_path: str,
    train_file: str,
    val_file: str,
    reward_fn_path: str,
    checkpoint_dir: str,
    train_batch_size: int = 4,
    ppo_mini_batch_size: int = 4,
    max_prompt_length: int = 512,
    max_response_length: int = 1200,
    ppo_max_token_len_per_gpu: int = 8192,
    lora_rank: int = 64,
    lora_alpha: int = 32,
    rollout_n: int = 4,
    rollout_gpu_mem_util: float = 0.6,
    total_training_steps: int = 3,
) -> list[str]:
    """Returns the full Hydra CLI arg list for `python3 -m verl.trainer.main_ppo`."""
    return [
        # ---- algorithm / data ----
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=False",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={train_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        # ---- model / LoRA / vision exclusion ----
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.model.lora_rank={lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.model.exclude_modules=.*visual.*",
        "actor_rollout_ref.model.use_fused_kernels=True",
        # ---- actor / Dr. GRPO fixes ----
        "actor_rollout_ref.actor.optim.lr=3e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
        # Fifth real bug (2026-08-10): both left False initially (matching
        # the 8-GPU LoRA example, where actor and rollout run on SEPARATE
        # GPU groups so offload isn't needed). On a single, colocated GPU
        # (actor + vLLM rollout sharing the same card), that meant the
        # actor's full weights+optimizer stayed permanently resident,
        # eating a huge fixed chunk regardless of rollout.
        # gpu_memory_utilization - which explained why lowering that
        # fraction from 0.5 to 0.25 produced BYTE-IDENTICAL OOM numbers
        # both times (confirmed by grepping the actual resolved CLI args,
        # not assumed): the fraction controls vLLM's slice of whatever is
        # ACTUALLY free, and the actor never freed anything. Real veRL
        # guidance (verl.readthedocs.io best-practices) confirms: with
        # offload enabled, gpu_memory_utilization of 0.8-0.9 is normal for
        # colocated setups - the opposite direction from where this was
        # first (wrongly) adjusted.
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        # Explicit bf16, not left to veRL's default. Verified via veRL's
        # real source (FSDPEngineConfig in verl/workers/engine.py,
        # 2026-08-10): the actual code-level default is fp32, not bf16 -
        # confirming a real risk, not a theoretical one (this project's
        # every other environment - the raw-HF proxy tests, Unsloth -
        # explicitly uses bf16; silently training in fp32 here would be a
        # real inconsistency, and use ~2x the memory of bf16, working
        # against everything already fixed above). Plain override of this
        # key is a documented, unresolved veRL bug ("Key 'model_dtype' is
        # not in struct" - Hydra struct-mode validation rejects it,
        # volcengine/verl#3243); using Hydra's own "+" prefix (force-add a
        # key even when struct mode would otherwise reject it) rather than
        # patching veRL's source code, which was the only workaround found
        # in that issue.
        # "+" prefix was wrong for THIS veRL version, not a guess left
        # uncorrected: a live run's error was explicit and self-diagnosing
        # ("Could not append to config. An item is already at
        # '...model_dtype'... Either remove + prefix... Or add a second +
        # ...") - meaning the "not in struct" bug from volcengine/verl#3243
        # doesn't apply to this build; the key IS already registered here,
        # so a plain override (no prefix) is correct.
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        # Tenth, most serious real bug found so far (2026-08-10): the real
        # root cause of the vision-weights-changed + gibberish-output
        # finding from an earlier run. Confirmed via veRL's actual current
        # dataclass (verl/workers/config/engine.py, FSDPEngineConfig -
        # the exact same class model_dtype above belongs to, confirmed by
        # directly reading its full field list, not assumed):
        # `use_orig_params: bool = False`. A first attempt guessed this
        # lived under actor_rollout_ref.model.fsdp_config, based on a
        # search result that turned out to describe older/different veRL
        # code - that attempt failed with a real, precise error
        # ("HFModelConfig.__init__() got an unexpected keyword argument
        # 'fsdp_config'"), which is what led to reading the actual current
        # dataclass directly instead of trusting the search summary a
        # second time. Since it's a genuine pre-declared field here
        # (confirmed sibling of model_dtype, which already works with a
        # plain override), no "+" prefix needed.
        #
        # Why this fixes the actual bug: real PyTorch/FSDP behavior when
        # use_orig_params=False - parameters flattened into the same FSDP
        # shard (FlatParameter) must share ONE requires_grad value - if
        # any trainable LoRA parameter ends up sharded together with the
        # (supposedly frozen, excluded) vision tower, the WHOLE group gets
        # forced to requires_grad=True, silently overriding
        # exclude_modules='.*visual.*' at the FSDP level regardless of
        # what the LoRA config itself says. Matches a closely-analogous
        # real veRL bug report (#3784: "Model Output Garbled at Step 2
        # when Enabling LoRA for GRPO/DAPO Training... Resolved When
        # Disabling LoRA") and the officially documented PyTorch fix for
        # LoRA+FSDP generally.
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=True",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
        f"actor_rollout_ref.actor.loss_scale_factor={max_response_length}",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]",
        # ---- rollout (vLLM) ----
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_mem_util}",
        # Eighth real bug (2026-08-10): without this, vLLM sized its KV
        # cache for Qwen2.5-VL's full native context (128,000 tokens,
        # confirmed via config.json earlier this project) instead of what
        # this run actually needs - real error: "To serve at least one
        # request with the model's max seq len (128000), 4.39 GiB KV cache
        # is needed... larger than the available KV cache memory (4.23
        # GiB)". Real precedent (verl.readthedocs.io best-practices)
        # confirms max_model_len should be derived from prompt+response
        # length, not left to default to the model's native max - that
        # derivation apparently isn't automatic in this build, so setting
        # it explicitly. max_prompt_length + max_response_length + margin.
        f"actor_rollout_ref.rollout.max_model_len={max_prompt_length + max_response_length + 100}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        # Ninth real fix (2026-08-10): was False. Real error was a
        # near-miss OOM (only 148 MiB short out of 39.49 GiB) during
        # actor_rollout_update_weights - syncing freshly-trained weights
        # back into vLLM after a real gradient step, a documented
        # peak-memory moment in colocated actor+rollout setups (both the
        # gathered actor weights and vLLM's own resident state briefly
        # coexist). Freeing vLLM's KV cache when not actively generating
        # frees exactly the room this moment needs, rather than broadly
        # lowering gpu_memory_utilization for the entire run (also nudged
        # down slightly above, but this is the more targeted lever).
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        # ---- reference policy ----
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={ppo_max_token_len_per_gpu}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.entropy_from_logits_with_chunking=True",
        # Same reasoning as the actor above - the reference policy needs
        # the same explicit bf16, both to avoid its own fp32-by-default
        # risk and to keep dtypes consistent with the actor during KL
        # computation. No "+" prefix needed here either (same
        # FSDPEngineConfig class, model_dtype is a genuine pre-declared
        # field).
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16",
        # Same use_orig_params fix as the actor above, same real reason:
        # the reference policy is also FSDP-wrapped and could suffer the
        # same requires_grad-uniformity problem independently of the
        # actor's own copy.
        "actor_rollout_ref.ref.fsdp_config.use_orig_params=True",
        # ---- trainer ----
        "trainer.balance_batch=True",
        "trainer.logger=[console]",
        "trainer.project_name=grpo-vlm-modality-shift",
        "trainer.experiment_name=qwen2_5_vl_3b_dry_run",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.save_freq={total_training_steps}",
        "trainer.test_freq=-1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={total_training_steps}",
        f"trainer.default_local_dir={checkpoint_dir}",
        "trainer.resume_mode=disable",
        "trainer.critic_warmup=0",
        # ---- custom reward ----
        f"custom_reward_function.path={reward_fn_path}",
        "custom_reward_function.name=compute_score",
    ]
