"""
Targeted, cheap verification of the sed patch applied to veRL's own
verl/models/transformers/qwen2_vl.py (configs/modal_app_verl.py) -
directly exercises the exact code path that crashed
("'BaseModelOutputWithPooling' object has no attribute 'mean'"), without
paying for the full vLLM+FSDP+Ray training pipeline again. If this
passes, the actual root cause is fixed; if it still fails, we know
immediately and cheaply, rather than waiting through another multi-minute
full dry-run cycle to find out.

Two checks:
  1. The patch actually landed in the built image (grep the file for the
     expected fixed line) - don't just assume a Docker build step ran
     correctly.
  2. The patched _get_input_embeds() function is called directly, with
     pixel_values=None (the exact text-only case that crashed), against
     the REAL loaded model - not a mock - and its output is inspected for
     sane shape/dtype, not just "didn't raise."

Usage: modal run scripts/verify_qwen2vl_patch_on_modal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

QWEN2VL_PATCH_FILE = "/opt/verl/verl/models/transformers/qwen2_vl.py"


@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=10 * 60,
)
def verify_patch() -> dict:
    import os

    os.environ["HF_HOME"] = MODEL_CACHE_DIR

    result = {"steps": []}

    # --- Check 1: the patch actually landed in the file ---
    with open(QWEN2VL_PATCH_FILE) as f:
        content = f.read()
    expected_fixed_snippet = (
        '0.0 * (image_embeds.last_hidden_state if hasattr(image_embeds, "last_hidden_state") '
        "else image_embeds).mean()"
    )
    patch_applied = expected_fixed_snippet in content
    result["patch_applied_in_file"] = patch_applied
    result["steps"].append(f"Patch present in qwen2_vl.py: {patch_applied}")
    if not patch_applied:
        result["status"] = "FAIL"
        result["error"] = "sed patch did not apply - check the exact original line still matches"
        return result

    # --- Check 2: call the actual patched function with pixel_values=None ---
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from verl.models.transformers.qwen2_vl import _get_input_embeds

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=MODEL_CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    result["steps"].append("Real model + processor loaded")

    # Real, new finding from the first run of this script: model.visual
    # doesn't exist directly - AttributeError before we even got back to
    # the original bug. Introspect the actual real attribute layout for
    # THIS installed transformers version rather than guess a second time.
    top_level_attrs = [a for a in dir(model) if not a.startswith("_")]
    result["model_top_level_attrs_with_visual_or_model"] = [
        a for a in top_level_attrs if "visual" in a.lower() or a == "model"
    ]
    result["steps"].append(
        f"model top-level attrs containing 'visual' or =='model': "
        f"{result['model_top_level_attrs_with_visual_or_model']}"
    )
    has_model_dot_visual = hasattr(model, "model") and hasattr(model.model, "visual")
    result["has_model_dot_visual"] = has_model_dot_visual
    result["steps"].append(f"hasattr(model.model, 'visual'): {has_model_dot_visual}")
    if has_model_dot_visual:
        result["steps"].append(f"type(model.model.visual): {type(model.model.visual).__name__}")

    text = "What is 15 + 27? Answer with just the number."
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
    result["steps"].append(f"Real text-only input built, input_ids shape={tuple(inputs['input_ids'].shape)}")

    try:
        inputs_embeds, attention_mask = _get_input_embeds(
            model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
        )
        result["steps"].append(
            f"_get_input_embeds() with pixel_values=None succeeded: "
            f"inputs_embeds shape={tuple(inputs_embeds.shape)}, dtype={inputs_embeds.dtype}, "
            f"has_nan={torch.isnan(inputs_embeds).any().item()}, "
            f"has_inf={torch.isinf(inputs_embeds).any().item()}"
        )
        result["inputs_embeds_shape"] = list(inputs_embeds.shape)
        result["has_nan"] = torch.isnan(inputs_embeds).any().item()
        result["has_inf"] = torch.isinf(inputs_embeds).any().item()
        result["status"] = "PASS" if not result["has_nan"] and not result["has_inf"] else "FAIL"
    except Exception as e:
        result["steps"].append(f"_get_input_embeds() FAILED: {type(e).__name__}: {e}")
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {e}"

    return result


@app.local_entrypoint()
def run():
    result = verify_patch.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    if "error" in result:
        print(f"\nERROR: {result['error']}")
    print(f"\nStatus: {result['status']}")
