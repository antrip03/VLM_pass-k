"""
Manipulation check (PLAN.md Phase 0 item 3, Phase 1 item 7): confirms the
vision encoder + projector weights are byte-identical before and after
LoRA/RL training - i.e., that "text-only RL" really did leave the vision
pathway completely untouched, not just an assumption.

VERIFIED (not guessed) parameter structure of Qwen2.5-VL-3B-Instruct,
confirmed via scripts/introspect_model_params.py on live Modal infra,
2026-08-10:
  visual.patch_embed.*    -> vision patch embedding
  visual.blocks.*         -> vision transformer blocks. Attention uses
                              fused attn.qkv naming (distinct from the
                              language decoder's separate q_proj/k_proj/
                              v_proj/o_proj) - but the MLP uses the SAME
                              leaf names as the language decoder's MLP
                              (gate_proj/up_proj/down_proj). A live
                              validation run proved this collision is
                              real: leaf-name-based LoRA target_modules
                              silently attached adapters to
                              visual.blocks.*.mlp.* too. See
                              build_lora_target_modules() below - it uses
                              fully-qualified paths, never bare leaf
                              names, specifically because of this.
  visual.merger.*         -> vision-language projector/merger
  model.embed_tokens.*    -> language token embeddings
  model.layers.{0..35}.*  -> language decoder transformer layers
    .self_attn.{q,k,v,o}_proj, .mlp.{gate,up,down}_proj,
    .input_layernorm, .post_attention_layernorm
  model.norm.*            -> final language decoder norm
  lm_head.*               -> language model head

So "vision pathway" = every parameter name containing "visual." - this is
the exclusion boundary used both here and by build_lora_target_modules()
below. Defense in depth: this check is the independent, authoritative
verification of what actually happened to the weights - it does not
assume the LoRA configuration was correct, it measures the outcome.

VERIFICATION STATUS: confirmed PASSING on live Modal infrastructure
(2026-08-10, scripts/validate_manipulation_check_on_modal.py), including
a REAL forward+backward+optimizer step (not just applying a LoRA config) -
all 390 vision parameters byte-identical before and after training, LoRA
correctly attached to exactly the 252 intended language-decoder modules
and zero vision modules. Getting here took four real, substantive bugs
found and fixed by this validation, not just cosmetic ones:
  1. hash_vision_state_dict crashed on bfloat16 (.numpy() doesn't support
     it) - fixed by viewing raw bytes as uint8 before conversion.
  2. is_vision_param's strict startswith("visual.") matched ZERO
     parameters once PEFT wrapped the model (adds a "base_model.model."
     prefix to every name) - silently made both hash_vision_state_dict's
     "after" snapshot AND check_lora_excludes_vision's scan vacuously
     empty (a false pass, not a real one). Fixed with a substring check.
  3. LoRA target_modules given as bare leaf names ("gate_proj", etc.)
     silently attached adapters to the vision blocks too, because
     Qwen2.5-VL's vision MLP happens to use the same leaf names as the
     language decoder's MLP - the single most serious bug, since it would
     have meant "text-only RL" wasn't actually text-only at all. Fixed by
     build_lora_target_modules() using fully-qualified paths.
  4. check_vision_unchanged reported every vision parameter as "changed"
     purely because the "before" snapshot (raw names) and "after"
     snapshot (PEFT-prefixed names) used different dict keys for the same
     tensors - not an actual weight difference. Fixed by normalizing hash
     keys to the stable "visual.*" suffix (_normalize_param_name).
"""

from __future__ import annotations

import hashlib

import torch

VISION_PREFIX = "visual."

# Verified via scripts/introspect_model_params.py on Qwen2.5-VL-3B-Instruct,
# 2026-08-10. Used only as a sanity floor (see _assert_found_vision_params),
# not an exact requirement - a different model/version could legitimately
# differ.
EXPECTED_VISION_PARAM_COUNT = 390


def is_vision_param(param_name: str) -> bool:
    """
    True if this parameter belongs to the vision encoder/projector.

    Uses a substring check, not a strict prefix check, because PEFT's
    get_peft_model() renames every parameter with a wrapper prefix
    (observed live: "base_model.model." prepended to everything). A
    strict startswith("visual.") check silently matched ZERO parameters
    on a PEFT-wrapped model - confirmed via a live validation run where
    this caused check_lora_excludes_vision to report clean=True while
    having scanned 0 vision params (a vacuous pass, not a real one), and
    hash_vision_state_dict to report an "after" snapshot with 0 vision
    params against a "before" snapshot with 390. Substring matching
    survives arbitrary wrapper prefixes.
    """
    return VISION_PREFIX in param_name


def _assert_found_vision_params(count: int, context: str) -> None:
    """
    Defensive floor: raises if zero vision params were matched, rather
    than letting a broken is_vision_param() silently produce a vacuous
    (falsely clean/unchanged) result - exactly the failure mode this
    module hit once already during its own validation. A loud error here
    is strictly better than a silent false pass.
    """
    if count == 0:
        raise RuntimeError(
            f"{context}: matched 0 vision parameters. This almost always "
            f"means is_vision_param() failed to match anything (e.g. an "
            f"unexpected parameter-naming wrapper), not that the model "
            f"genuinely has no vision parameters."
        )


def _normalize_param_name(param_name: str) -> str:
    """
    Returns the stable "visual.*" suffix of a parameter name, discarding
    any wrapper prefix before it (e.g. PEFT's "base_model.model."). A live
    validation run proved this normalization is necessary, not
    theoretical: comparing a snapshot taken before LoRA wrapping (raw
    names like "visual.blocks.0...") against one taken after wrapping
    (PEFT-prefixed names like "base_model.model.visual.blocks.0...")
    reported every single vision parameter as "changed" - purely a dict-
    key mismatch from the wrapper prefix, not an actual weight
    difference. Using this normalized suffix as the hash-dict key instead
    of the raw name makes the same underlying tensor map to the same key
    regardless of what wraps it.
    """
    idx = param_name.index(VISION_PREFIX)  # is_vision_param already guarantees this exists
    return param_name[idx:]


def hash_vision_state_dict(model) -> dict[str, str]:
    """
    Returns {normalized_name: sha256_hex} for every vision-pathway
    parameter in the model, keyed by the stable "visual.*" suffix (see
    _normalize_param_name) so snapshots remain comparable across a PEFT-
    wrapping boundary. Hashing (rather than storing full tensors) keeps
    this cheap and avoids holding two full copies of the vision weights
    in memory when comparing a before/after snapshot.
    """
    hashes: dict[str, str] = {}
    for name, param in model.named_parameters():
        if is_vision_param(name):
            # .numpy() fails directly on bfloat16 (not a native NumPy
            # dtype - confirmed by a live run: "TypeError: Got unsupported
            # ScalarType BFloat16"). Reinterpreting the raw memory as
            # uint8 first sidesteps that entirely, and is arguably more
            # correct anyway: it hashes the literal bit pattern with zero
            # numeric conversion, matching this function's "byte-
            # identical" contract exactly, regardless of the parameter's
            # actual dtype.
            tensor_bytes = (
                param.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
            )
            hashes[_normalize_param_name(name)] = hashlib.sha256(tensor_bytes).hexdigest()
    _assert_found_vision_params(len(hashes), context="hash_vision_state_dict")
    return hashes


def check_vision_unchanged(
    hashes_before: dict[str, str], hashes_after: dict[str, str]
) -> dict:
    """
    Compares two hash snapshots (e.g. taken before and after a LoRA
    training run). `unchanged` must be True and `changed_params` must be
    empty for the manipulation check to pass. This should be reported
    explicitly for every seed (PLAN.md Phase 1 item 7), not just checked
    once and assumed to hold.
    """
    before_keys = set(hashes_before)
    after_keys = set(hashes_after)
    if before_keys != after_keys:
        return {
            "unchanged": False,
            "error": "vision parameter set differs between snapshots",
            "only_before": sorted(before_keys - after_keys),
            "only_after": sorted(after_keys - before_keys),
        }

    changed = [name for name in hashes_before if hashes_before[name] != hashes_after[name]]
    return {
        "unchanged": len(changed) == 0,
        "changed_params": changed,
        "total_vision_params_checked": len(hashes_before),
    }


_LANGUAGE_ATTN_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
_LANGUAGE_MLP_PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]


def build_lora_target_modules(model) -> list[str]:
    """
    Returns fully-qualified module paths (e.g.
    "model.layers.5.self_attn.q_proj") for every attention/MLP Linear
    layer in the LANGUAGE decoder only, for use as PEFT LoraConfig's
    target_modules - suitable for reuse by Step 5's training script, not
    just this validation.

    Deliberately does NOT use leaf names like "gate_proj" as
    target_modules (PEFT's default matching style). A live validation run
    proved that's unsafe here: Qwen2.5-VL's vision blocks use the SAME
    leaf names as the language decoder's MLP (gate_proj/up_proj/
    down_proj) - confirmed by LoRA adapters actually attaching to
    "base_model.model.visual.blocks.0.mlp.gate_proj..." when leaf-name
    matching was used, despite the vision attention layers using a
    different (fused qkv) naming that made it look safe from partial
    inspection. Fully-qualified paths anchored to "model.layers.{i}." are
    unambiguous: a vision module's path always starts with "visual.", so
    it can never equal or end with a "model.layers.N...." string
    regardless of PEFT's exact match/endswith semantics.

    num_hidden_layers is read from the model's own config rather than
    hardcoded, so this stays correct if reused for a differently-sized
    model in Phase 2 (PLAN.md Section 3).
    """
    num_layers = model.config.num_hidden_layers
    targets = []
    for i in range(num_layers):
        for proj in _LANGUAGE_ATTN_PROJECTIONS:
            targets.append(f"model.layers.{i}.self_attn.{proj}")
        for proj in _LANGUAGE_MLP_PROJECTIONS:
            targets.append(f"model.layers.{i}.mlp.{proj}")
    return targets


def check_lora_excludes_vision(model_with_lora) -> dict:
    """
    After applying a PEFT LoRA config, confirms no trainable (LoRA)
    parameter was attached anywhere under the vision prefix - a
    complementary check to check_vision_unchanged: this one catches a
    misconfigured target_modules list *before* training even starts
    (fails fast), rather than only detecting the consequence afterward.

    Reports total_vision_params_seen alongside the clean/trainable_vision_
    params verdict, and raises if it's 0 - otherwise "clean: True" could
    mean "genuinely verified no trainable vision params" OR "silently
    matched nothing at all and verified nothing," which is exactly the
    vacuous-pass failure mode this module hit once already (see
    is_vision_param's docstring).
    """
    total_vision_params_seen = 0
    trainable_vision_params = []
    for name, param in model_with_lora.named_parameters():
        if is_vision_param(name):
            total_vision_params_seen += 1
            if param.requires_grad:
                trainable_vision_params.append(name)

    _assert_found_vision_params(
        total_vision_params_seen, context="check_lora_excludes_vision"
    )

    return {
        "clean": len(trainable_vision_params) == 0,
        "trainable_vision_params": trainable_vision_params,
        "total_vision_params_seen": total_vision_params_seen,
    }
