"""
One-off debug script: discovers Qwen2.5-VL-3B-Instruct's real named-
parameter structure (which prefixes are vision encoder/projector vs.
language decoder), so src/controls/manipulation_check.py can filter on
verified real names instead of a guess. Not part of the production
pipeline - run once, use the output to write the real check, then this
script isn't needed again unless the model changes.

Usage: modal run scripts/introspect_model_params.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache


@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=5 * 60,
)
def introspect() -> dict:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_DIR,
    )

    # Top-level named children of the model (broad structure).
    top_level = [name for name, _ in model.named_children()]

    # All unique first-two-segment prefixes across every parameter name,
    # e.g. "visual.blocks" or "model.layers" - enough to identify which
    # prefixes correspond to vision vs. language weights without dumping
    # every single one of the ~300+ parameter names.
    prefixes = set()
    for name, _ in model.named_parameters():
        parts = name.split(".")
        prefixes.add(".".join(parts[:2]))

    # A few concrete full parameter names, to sanity-check against
    # transformers' actual naming (not just prefixes).
    sample_names = [name for name, _ in model.named_parameters()][:5]
    sample_names_tail = [name for name, _ in model.named_parameters()][-5:]

    # Every parameter name within decoder layer 0 specifically - needed to
    # get LoRA target_modules exactly right (attention proj names weren't
    # visible in the head/tail samples above).
    layer0_names = [
        name for name, _ in model.named_parameters() if name.startswith("model.layers.0.")
    ]

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "top_level_children": top_level,
        "unique_prefixes": sorted(prefixes),
        "sample_names_head": sample_names,
        "sample_names_tail": sample_names_tail,
        "layer0_names": layer0_names,
        "total_params_millions": round(total_params / 1e6, 1),
    }


@app.local_entrypoint()
def introspect_main():
    result = introspect.remote()
    print("top_level_children:", result["top_level_children"])
    print("\nunique_prefixes:")
    for p in result["unique_prefixes"]:
        print(" ", p)
    print("\nsample_names_head:", result["sample_names_head"])
    print("sample_names_tail:", result["sample_names_tail"])
    print("\nlayer0_names:")
    for n in result["layer0_names"]:
        print(" ", n)
    print("\ntotal_params_millions:", result["total_params_millions"])
