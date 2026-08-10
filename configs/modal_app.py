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

VERIFICATION STATUS:
  Confirmed PASSING on live Modal infrastructure (A10G) on 2026-08-10:
  GPU detected (NVIDIA A10), image built cleanly, model + processor loaded
  in ~170s (dominated by the first-time ~7-8GB weight download), both
  text-mode and image-mode generation succeeded and both answered the
  smoke-test arithmetic question correctly. Run: https://modal.com/apps/guneesh-g/main/ap-DKBKc0P32djsjB04tiYy2J

  One real bug was caught and fixed by this run: qwen_vl_utils.vision_process
  imports torchvision internally, which was missing from the original pins
  (only torch was listed) - added torchvision==0.20.1 (matching torch==2.5.1
  per PyTorch's compatibility matrix) to fix it. A separate, cosmetic issue
  (Windows console can't render the Modal CLI's Unicode checkmark output)
  was worked around by running with PYTHONIOENCODING=utf-8 - not a bug in
  this file, but worth knowing if `modal run` appears to crash immediately
  with a charmap encoding error on Windows.

  Still not independently re-verified: the exact package version pins
  beyond what this run exercised, and the A100-40GB GPU path (unused by
  this smoke test, reserved for Step 5's training script).
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
    "torchvision==0.20.1",  # must match torch==2.5.1 per PyTorch's compatibility
    # matrix; required by qwen_vl_utils.vision_process, which imports it
    # internally - missing on the first run, caught by the live smoke test.
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
