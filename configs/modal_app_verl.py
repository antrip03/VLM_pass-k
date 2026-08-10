"""
Separate Modal environment for veRL-based training (PLAN.md Section 3
Step 5), kept deliberately independent from configs/modal_app.py.

Why a second, separate image rather than upgrading the existing one:
veRL requires torch>=2.10.0, torchvision>=0.23.0, vllm==0.17.0+,
flash-attn>=2.8.1, CUDA>=12.8 - all newer than the pins already validated
across Steps 1-4 (torch==2.5.1, torchvision==0.20.1). Upgrading the
shared image would force re-validating everything already proven working
(smoke test, rendering, manipulation check, headroom check) against an
untested stack. Steps 1-4 don't need veRL's dependencies at all, so there
is no reason to couple them.

Base image: verlai/verl:vllm011.latest, a prebuilt Docker image from
veRL's own DockerHub (confirmed via https://verl.readthedocs.io/en/latest/start/install.html) -
Docker is documented as the RECOMMENDED install path for veRL, not plain
pip, and building up from debian_slim + pip (as configs/modal_app.py
does) would mean manually replicating veRL's CUDA/vLLM/flash-attn
toolchain ourselves - a much larger, riskier surface than using their
own maintained image.

IMPORTANT, learned the hard way (see VERIFICATION STATUS below): despite
the name, this image provides the DEPENDENCY STACK (torch, vLLM, CUDA,
flash-attn) - it does NOT include verl itself pre-installed. The install
docs actually document a two-step process (pull this image, then clone +
`pip install --no-deps -e .` verl inside it); a first attempt only did
step one. The image build below now does both.

VERIFICATION STATUS: two real, structural bugs found and fixed via live
Modal runs before this got here, neither a mistake in our own code so
much as incorrect assumptions about this specific base image:
  1. First attempt crashed every container on startup - this image ships
     a baked-in ENTRYPOINT that auto-launches vLLM's OpenAI-compatible API
     server, hijacking whatever command Modal injects to run our function.
     Fixed with setup_dockerfile_commands=["ENTRYPOINT []"].
  2. Second attempt got past container startup but hit
     "ModuleNotFoundError: No module named 'verl'" - see IMPORTANT above.
     Fixed by adding the git-clone + editable-install step.

Third attempt, with both fixes applied, PASSED on live Modal infrastructure
(2026-08-10): GPU detected (NVIDIA A10), torch 2.8.0+cu128 (note: lower
than the torch>=2.10.0 seen in verl's pip package metadata during
planning - this specific Docker tag bundles an older-but-working torch;
trust what's actually installed and running over pip metadata read
separately), verl imported correctly from the git-clone path
(/opt/verl/verl/__init__.py), verl.trainer.main_ppo imported OK, and the
full Hydra CLI (--help) ran cleanly (exit code 0), dumping verl's
complete default config schema.

Not yet done: a real training run (this only confirmed the CLI parses
its own arguments, not that training actually executes), the Parquet
data conversion, and the custom reward function wiring our own
src/metrics/answer_extraction.py logic (needed because verl's own
built-in gsm8k.py reward function has the same "#### only" extraction
gap already found and fixed in Step 4 - see that module's docstring).
"""

import modal

APP_NAME = "grpo-vlm-modality-shift-verl"

image = modal.Image.from_registry(
    "verlai/verl:vllm011.latest",
    # No add_python: this image already ships Python + torch + vllm + verl
    # pre-installed. Passing add_python would risk adding a SECOND,
    # separate Python installation that can't see any of those
    # pre-installed packages, breaking the whole point of using this base
    # image - confirmed by reading Image.from_registry's actual docstring
    # before using it, not assumed.
    setup_dockerfile_commands=["ENTRYPOINT []"],
    # A first attempt without this crashed every container on startup:
    # "api_server.py: error: unrecognized arguments: -u -R
    # --check-hash-based-pycs never -m modal._container_entrypoint".
    # This image is serving-oriented and ships a baked-in ENTRYPOINT that
    # auto-launches vLLM's OpenAI-compatible API server on container
    # start, hijacking whatever command Modal tries to inject to run our
    # actual Python function - it force-fed Modal's own startup arguments
    # to vLLM's server script, which of course didn't recognize them.
    # Clearing the entrypoint restores normal shell behavior so Modal can
    # control container startup the way it needs to.
).run_commands(
    # This base image provides the DEPENDENCY STACK (torch, vLLM, CUDA,
    # flash-attn) but does NOT include verl itself pre-installed - a
    # second real finding, caught the same way: a first attempt without
    # this step got "ModuleNotFoundError: No module named 'verl'" after
    # the entrypoint fix let the container actually start. The install
    # docs (verl.readthedocs.io/en/latest/start/install.html) show this
    # exact two-step process: pull the dependency image, THEN clone +
    # editable-install verl inside it. Baked into the image at build time
    # (not run per-container) so it's cached, matching the documented
    # "pip3 install --no-deps -e ." command exactly.
    "git clone https://github.com/volcengine/verl /opt/verl",
    "cd /opt/verl && pip3 install --no-deps -e .",
).add_local_python_source("src", "configs")

app = modal.App(APP_NAME, image=image)

model_cache = modal.Volume.from_name("grpo-vlm-model-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/cache/huggingface"

SMOKE_TEST_GPU = "A10G"


@app.function(
    image=image,
    gpu=SMOKE_TEST_GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    timeout=15 * 60,
)
def verl_smoke_test() -> dict:
    """
    Minimal check, deliberately not a real training run yet:
      1. GPU visible inside this Docker-based image
      2. `import verl` succeeds
      3. veRL's version / package location, for the record
      4. The main_ppo CLI module is importable (confirms the entry point
         referenced by every example script actually exists in this image)
    """
    import subprocess

    result = {"steps": []}

    import torch

    assert torch.cuda.is_available(), "CUDA not available inside the container"
    result["steps"].append(f"GPU OK: {torch.cuda.get_device_name(0)}")
    result["steps"].append(f"torch version: {torch.__version__}")

    import verl

    result["steps"].append(f"verl imported OK, location: {verl.__file__}")

    try:
        import verl.trainer.main_ppo  # noqa: F401

        result["steps"].append("verl.trainer.main_ppo imported OK")
    except Exception as e:
        result["steps"].append(f"verl.trainer.main_ppo import FAILED: {e}")
        result["status"] = "FAIL"
        return result

    # Confirm the CLI entry point itself runs (--help should exit 0
    # without needing any real config).
    proc = subprocess.run(
        ["python3", "-m", "verl.trainer.main_ppo", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    result["steps"].append(f"CLI --help exit code: {proc.returncode}")
    result["cli_help_stdout_tail"] = proc.stdout[-1000:]
    result["cli_help_stderr_tail"] = proc.stderr[-1000:]

    result["status"] = "PASS" if proc.returncode == 0 else "FAIL"
    return result


@app.local_entrypoint()
def smoke_test():
    result = verl_smoke_test.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    if "cli_help_stdout_tail" in result:
        print(f"\n--- CLI --help stdout (tail) ---\n{result['cli_help_stdout_tail']}")
    if result.get("cli_help_stderr_tail"):
        print(f"\n--- CLI --help stderr (tail) ---\n{result['cli_help_stderr_tail']}")
    print(f"\nStatus: {result['status']}")
