"""
Separate Modal environment for Unsloth - a comparison probe against veRL
(configs/modal_app_verl.py), requested after the "is Unsloth actually 2x
faster for OUR workload" discussion. Kept independent from
configs/modal_app.py and configs/modal_app_verl.py for the same reason
those two are independent of each other: different, often mutually
incompatible dependency stacks. This is a comparison probe, not yet a
committed replacement for anything.

Real, verified install requirements (checked before writing this,
2026-08-10, via Unsloth's own docs - not memorized/guessed): base install
is `pip install unsloth`; Qwen2.5-VL specifically needs transformers
built from source per Unsloth's docs, plus `qwen-vl-utils[decord]`.

Deliberately NOT pinning exact torch/vllm versions here (unlike
modal_app.py's tight pins) - Unsloth and vLLM both have their own
internal version compatibility requirements that shift between releases,
and guessing exact pins risks a dependency-resolution conflict on the
first attempt. Letting pip's resolver pick compatible versions is more
likely to succeed than a guessed pin, matching how veRL's own official
Docker image was used specifically to avoid this same class of problem.
If this environment becomes a committed part of the pipeline (not just a
comparison probe), pin the resolved versions then for reproducibility -
premature here.
"""

import modal

APP_NAME = "grpo-vlm-modality-shift-unsloth"

# Same base model ID as every other comparison in this project
# (configs/modal_app.py, configs/modal_app_verl.py) - not Unsloth's own
# re-uploaded variant, to keep this a comparison of the TOOLING, not a
# different starting checkpoint.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        # Second real bug found here (2026-08-10): a first attempt used
        # plain `pip install unsloth unsloth_zoo vllm trl transformers ...`
        # (all separate top-level constraints). unsloth_zoo declares NO
        # vllm version constraint at all in its own package metadata, so
        # pip's resolver had nothing to anchor to and landed on vllm 0.26.0
        # - too new for that unsloth_zoo release's internal
        # `unsloth_zoo/vllm_lora_worker_manager.py`, which imports
        # `vllm.lora.models` (a module path that newer vllm no longer
        # exposes there) -> ModuleNotFoundError at `from unsloth import
        # FastVisionModel` itself (an unconditional import chain, confirmed
        # via a real open, unresolved upstream GitHub issue hitting the
        # identical crash even for users who never wanted vLLM at all).
        # Checked Unsloth's own docs and install.sh directly (not
        # guessed): neither pins an exact vllm version either - their own
        # documented command is `uv pip install unsloth vllm
        # --torch-backend=auto`, using uv's resolver, installing
        # unsloth+vllm together with nothing else competing for
        # resolution. This second attempt matches that as closely as
        # possible (uv, not pip; everything resolved together in one pass,
        # not layered pip_install calls that could re-resolve shared deps
        # inconsistently) - a genuine best-effort at the officially
        # documented path, not a guessed version pin (none exists to copy).
        #
        # Third real bug (2026-08-10): the doc-matched command above
        # (including --torch-backend=auto) DID fix the vllm.lora.models
        # crash - vllm resolved to 0.23.0, imports worked - but installed
        # torch==2.11.0+cpu, a CPU-only build. Root cause: --torch-backend=
        # auto detects hardware AT IMAGE-BUILD TIME to pick a matching
        # PyTorch backend (cpu/cu121/cu124/...), but Modal builds images
        # without a GPU attached (the GPU only appears once the function
        # actually runs) - so it correctly saw "no GPU visible right now"
        # and picked the CPU wheel, which then can't use the real GPU at
        # runtime. Structural mismatch between that flag's assumption
        # (install-time GPU visibility) and Modal's two-phase build/run
        # architecture, not a version problem. Fixed by dropping
        # --torch-backend=auto - configs/modal_app.py and
        # configs/modal_app_verl.py both already prove a plain torch
        # install (no special backend flag) correctly resolves a
        # CUDA-capable wheel on Modal.
        "uv pip install --system unsloth vllm "
        "'transformers>=4.49.0' accelerate peft trl "
        "'qwen-vl-utils[decord]==0.0.8' pillow huggingface_hub datasets",
    )
    .add_local_python_source("src", "configs")
)

app = modal.App(APP_NAME, image=image)

model_cache = modal.Volume.from_name("grpo-vlm-model-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/cache/huggingface"

GPU = "A100-40GB"  # matches the veRL-proxy comparison numbers exactly
