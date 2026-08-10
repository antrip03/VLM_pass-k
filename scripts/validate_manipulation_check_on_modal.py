"""
Validates src/controls/manipulation_check.py against the real model on
live Modal infrastructure. Goes beyond just applying a LoRA config -
also runs one real dummy forward+backward+optimizer step (on a tiny
synthetic text batch) and re-checks afterward, since the actual concern
(PLAN.md Phase 1 item 7) is whether TRAINING leaves vision untouched, not
just whether wrapping the model in a PEFT config does.

Usage: modal run scripts/validate_manipulation_check_on_modal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.modal_app import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

# NOTE: target_modules is no longer a hardcoded leaf-name list here - it's
# built dynamically per-model via manipulation_check.build_lora_target_modules(),
# using fully-qualified paths. A first attempt at this validation used bare
# leaf names ("q_proj", "gate_proj", ...) and that silently attached LoRA
# adapters to the vision blocks too (they share MLP leaf names with the
# language decoder) - see build_lora_target_modules()'s docstring for the
# full story. Do not revert to bare leaf names.


@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=5 * 60,
)
def validate_manipulation() -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.controls.manipulation_check import (
        build_lora_target_modules,
        check_lora_excludes_vision,
        check_vision_unchanged,
        hash_vision_state_dict,
    )

    result = {"steps": []}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    result["steps"].append("Model loaded")

    hashes_before = hash_vision_state_dict(model)
    result["steps"].append(f"Hashed {len(hashes_before)} vision params (before LoRA)")

    target_modules = build_lora_target_modules(model)
    result["steps"].append(
        f"Built {len(target_modules)} fully-qualified LoRA target modules "
        f"(language decoder only)"
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    result["steps"].append("Applied LoRA config")

    exclusion_check = check_lora_excludes_vision(model)
    result["lora_excludes_vision"] = exclusion_check
    result["steps"].append(f"LoRA-excludes-vision check: {exclusion_check['clean']}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    result["trainable_param_pct"] = round(100 * trainable_params / total_params, 3)
    result["steps"].append(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({result['trainable_param_pct']}%)"
    )

    # Real dummy forward + backward + optimizer step, text-only, to confirm
    # actual training (not just wrapping) leaves vision untouched.
    text_messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is 2 + 2?"}]}
    ]
    prompt = processor.apply_chat_template(
        text_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    result["steps"].append(f"Dummy forward+backward+optimizer step OK (loss={loss.item():.4f})")

    hashes_after = hash_vision_state_dict(model)
    unchanged_check = check_vision_unchanged(hashes_before, hashes_after)
    result["vision_unchanged"] = unchanged_check
    result["steps"].append(f"Vision-unchanged check after real training step: {unchanged_check['unchanged']}")

    result["status"] = (
        "PASS"
        if exclusion_check["clean"] and unchanged_check["unchanged"]
        else "FAIL"
    )
    return result


@app.local_entrypoint()
def validate():
    result = validate_manipulation.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    print(f"\nlora_excludes_vision: {result['lora_excludes_vision']}")
    print(f"vision_unchanged: {result['vision_unchanged']}")
    print(f"\nStatus: {result['status']}")
