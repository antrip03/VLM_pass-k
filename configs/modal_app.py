"""
Modal environment for the GRPO-VLM modality-shift project.

Covers PLAN.md Step 1: pinned container image, GPU config, and the
Qwen2.5-VL-3B-Instruct smoke-test function (PLAN.md Section 3, Phase 0).

WHAT THIS FILE DOES NOT DO YET (later steps, per PLAN.md Section 3):
  - Headroom pre-check (pass@1 vs pass@16 on real problems)   -> Step 4
  - Manipulation check (vision-tower weight diff)               -> Step 3
  - Real GSM8K-V image rendering                                -> Step 2
  - LoRA / Dr. GRPO training                                    -> Step 5
This file only proves the environment + model loading + generation
pathway works, in both text and (synthetic) image mode.

VERIFICATION STATUS - READ BEFORE TRUSTING THIS FILE:
  This was written without a Modal account/credentials available in the
  authoring environment, so it has NOT been executed against live Modal
  infrastructure. Syntax was checked locally (py_compile) but nothing
  beyond that. Before relying on it:
    1. Run `modal run configs/modal_app.py` yourself and confirm smoke_test()
       actually passes on real hardware.
    2. Double-check the package pins below against what's actually
       available/compatible today - they reflect versions the author is
       confident existed as of their training data (cutoff ~Jan 2026);
       transformers' Qwen-VL support moves fast, confirm before trusting.
    3. Double-check the Modal GPU string identifiers ("A10G", "A100-40GB")
       against Modal's current docs - naming has changed before.
    4. Double-check the transformers class name
       (Qwen2_5_VLForConditionalGeneration) and the qwen_vl_utils API
       (process_vision_info) against the installed package versions.
  Treat this as a well-reasoned first draft, not a validated artifact.
"""

import modal

APP_NAME = "grpo-vlm-modality-shift"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# ---------------------------------------------------------------------------
# Container image - pinned for reproducibility (PLAN.md Section 7 emphasizes
# reproducibility; the same principle applies to the environment itself, not
# just the training algorithm).
# ---------------------------------------------------------------------------
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==4.49.0",
    "accelerate==1.2.1",
    "peft==0.14.0",
    "qwen-vl-utils==0.0.8",
    "pillow==11.0.0",
    "huggingface_hub==0.27.0",
    "einops==0.8.0",
)

app = modal.App(APP_NAME, image=image)

# Volume to cache downloaded HF model weights across container restarts -
# avoids re-downloading ~6GB every time a function runs. Persists between
# calls; committed explicitly after a fresh download (see smoke_test below).
model_cache = modal.Volume.from_name("grpo-vlm-model-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/cache/huggingface"

# GPU allocation, per README.md / PLAN.md Section 9:
#   A10G       - cheap, Phase 0 smoke test + pilot only
#   A100-40GB  - required for Phase 1 (3B full run) and any 7B work
SMOKE_TEST_GPU = "A10G"
FULL_RUN_GPU = "A100-40GB"  # not used in this file yet - reserved for Step 5


@app.function(
    gpu=SMOKE_TEST_GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=15 * 60,
)
def smoke_test() -> dict:
    """
    Phase 0 smoke test (PLAN.md Section 3, Phase 0, items 1 and parts of
    5-7). Confirms:
      - GPU is visible and usable inside the container
      - Qwen2.5-VL-3B-Instruct loads correctly (weights cached to a Volume
        after the first run)
      - A minimal TEXT-mode generation runs end to end
      - A minimal IMAGE-mode generation runs end to end, using a synthetic
        on-the-fly image (NOT the real GSM8K-V rendering pipeline - that is
        built in Step 2; this only proves the vision pathway itself works)

    Returns a dict of step-by-step results so failures are easy to localize
    rather than a single silent pass/fail.
    """
    import time

    import torch
    from PIL import Image, ImageDraw
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    result = {"steps": [], "status": "FAIL"}

    # 1. GPU check
    assert torch.cuda.is_available(), "CUDA not available inside the container"
    result["steps"].append(f"GPU OK: {torch.cuda.get_device_name(0)}")

    # 2. Load model + processor
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=MODEL_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    result["steps"].append(f"Model + processor loaded in {time.time() - t0:.1f}s")
    model_cache.commit()  # persist any newly-downloaded weights to the Volume

    # 3. Text-mode generation
    text_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is 15 + 27? Answer with just the number."}
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        text_messages, tokenize=False, add_generation_prompt=True
    )
    text_inputs = processor(text=[text_prompt], return_tensors="pt").to("cuda")
    t0 = time.time()
    text_out = model.generate(**text_inputs, max_new_tokens=32)
    text_decoded = processor.batch_decode(
        text_out[:, text_inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]
    result["steps"].append(
        f"Text-mode generation OK ({time.time() - t0:.1f}s): {text_decoded!r}"
    )

    # 4. Image-mode generation - synthetic image, NOT the real GSM8K-V
    #    rendering pipeline (that's Step 2). Only confirms the vision
    #    pathway (encoder -> projector -> decoder) works at all.
    img = Image.new("RGB", (400, 120), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "What is 15 + 27?", fill="black")

    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {
                    "type": "text",
                    "text": "Answer the question in the image with just the number.",
                },
            ],
        }
    ]
    image_prompt = processor.apply_chat_template(
        image_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs_vision, _ = process_vision_info(image_messages)
    image_inputs = processor(
        text=[image_prompt], images=image_inputs_vision, return_tensors="pt"
    ).to("cuda")
    t0 = time.time()
    image_out = model.generate(**image_inputs, max_new_tokens=32)
    image_decoded = processor.batch_decode(
        image_out[:, image_inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]
    result["steps"].append(
        f"Image-mode generation OK ({time.time() - t0:.1f}s): {image_decoded!r}"
    )

    result["status"] = "PASS"
    return result


@app.local_entrypoint()
def main():
    """Run via: modal run configs/modal_app.py"""
    result = smoke_test.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    print(f"\nStatus: {result['status']}")
