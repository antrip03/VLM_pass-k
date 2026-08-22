"""
Phase 1b Step 4: the RANDOMISED-REWARD CONTROL.

THE OBJECTION THIS ANSWERS
--------------------------
Shao et al. (arXiv 2506.10947) showed that on Qwen models RLVR yields
large gains even when the reward is uncorrelated with correctness
(+21.4% on MATH-500 from a purely random reward, against +29.1% from the
ground-truth reward), and that the effect "generally fails" on Llama3 or
OLMo2. Their explicit warning: "RLVR research validated solely on Qwen
models might not generalize."

This project trains Qwen2.5-VL-3B and reports Delta_text = +0.0587. A
reviewer who knows that paper will ask whether that gain came from the
correctness signal at all. Answering by analogy ("different benchmark,
different model") is weak. This control answers it directly: run the same
training with the reward severed from correctness and show the gain does
not appear.

STEP-MATCHED AT 250, NOT 250-vs-467
-----------------------------------
The control trains to step 250 and is compared against the REAL run's
step-250 checkpoint - not against the final step-467 result. Comparing
random@250 with real@467 would confound reward-validity with training
length, which is exactly the asymmetry Shao et al.'s own design avoids:
they trained every reward condition (ground truth, random, format,
incorrect label, majority vote) for the SAME 300 steps on a shared axis,
and where they ran longer (650 steps) it was the RANDOM condition they
extended, never the reference.

Step 250 is chosen because (a) a real checkpoint exists there
(save_freq=25), and (b) it is comfortably past the window in which
spurious-reward effects emerge - Shao et al. see gains "within the first
50 steps" for most spurious rewards, and after ~100 steps for random
rewards specifically on smaller models. Stopping at 150-200 would risk
halting right at the onset and reporting an uninformative null.

The paper must therefore state the claim at step 250, and say plainly
that extending the control to 467 was not done.

CONDITIONS T AND E, NOT T ALONE
-------------------------------
T establishes that the training signal produced a real gain. E
establishes it for the modality-shifted measurement that is the paper's
actual headline (Delta_pixel). A T-only control leaves a reviewer able to
say "the text gain is controlled, but the claim is about image transfer,
which is not" - and with no rebuttal round at a workshop, that objection
would stand unanswered.

Affordable because the control needs pass@1, not pass@k: n=16 suffices
where the headline used n=128, an 8x reduction on the expensive
condition.

    # 1. train (~3.5-4hr on A100-40GB at the measured ~55s/step)
    modal run scripts/run_random_reward_control_on_modal.py::train_random_reward

    # 2. evaluate both models at step 250 (~1hr)
    modal run scripts/run_random_reward_control_on_modal.py::evaluate

    # 3. compare
    modal run scripts/run_random_reward_control_on_modal.py::analyze_control
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

DATA_DIR = "/data"
CONTROL_CKPT_DIR = "/control_checkpoints"
RESULTS_DIR = "/results"

data_volume = modal.Volume.from_name("grpo-vlm-phase1-data", create_if_missing=True)
control_ckpt_volume = modal.Volume.from_name(
    "grpo-vlm-randomreward-checkpoints", create_if_missing=True
)
results_volume = modal.Volume.from_name("grpo-vlm-phase1b-results", create_if_missing=True)

TRAIN_FILE = f"{DATA_DIR}/gsm8k_train.parquet"
VAL_FILE = f"{DATA_DIR}/gsm8k_test.parquet"

HF_CKPT_REPO = "GunGG4/grpo-vlm-phase1-checkpoints"  # the REAL run's checkpoints
CONTROL_STEP = 250
EXPERIMENT_NAME = "qwen2_5_vl_3b_randomreward_control"

TRAINING_TIMEOUT_SECONDS = 8 * 60 * 60
EVAL_GPU = "A10G"
EVAL_N = 16  # pass@1 is the question; n=128 would be 8x cost for no extra answer
EVAL_PROBLEMS = 50
EVAL_CONDITIONS = ["T", "E"]


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={
        MODEL_CACHE_DIR: model_cache,
        DATA_DIR: data_volume,
        CONTROL_CKPT_DIR: control_ckpt_volume,
    },
    timeout=TRAINING_TIMEOUT_SECONDS,
)
def train_random_reward(total_training_steps: int = CONTROL_STEP, wandb_mode_disabled: bool = False) -> dict:
    """
    Train to step 250 with the randomised reward.

    EVERY hyperparameter other than the reward function, the checkpoint
    directory, the step count and the experiment name is left exactly as
    the real Phase 1 run - verified by diffing build_args() output, which
    shows those four keys and nothing else changing. Any further
    divergence would confound this control with an unrelated change.
    """
    import collections
    import os
    import re
    import subprocess
    import time

    import src as _src

    from src.training.phase1_config import build_args

    src_root = Path(_src.__file__).resolve().parent
    reward_fn_path = src_root / "training" / "reward_fn_random.py"
    if not reward_fn_path.exists():
        raise RuntimeError(f"Randomised reward function not found at {reward_fn_path}")

    args = build_args(
        model_path=MODEL_ID,
        train_file=TRAIN_FILE,
        val_file=VAL_FILE,
        reward_fn_path=str(reward_fn_path),
        checkpoint_dir=CONTROL_CKPT_DIR,
        total_training_steps=total_training_steps,
        checkpoint_every=25,
        # No baseline validation pass: it costs ~8 min and measures the
        # BASE model, which Phase 1 already measured on the same data.
        val_before_train=False,
        experiment_name=EXPERIMENT_NAME,
    )

    env = os.environ.copy()
    env["HF_HOME"] = MODEL_CACHE_DIR
    if wandb_mode_disabled:
        env["WANDB_MODE"] = "disabled"

    print(f"=== RANDOM-REWARD CONTROL: training to step {total_training_steps} ===", flush=True)
    print(f"reward_fn: {reward_fn_path}", flush=True)
    print(f"checkpoints: {CONTROL_CKPT_DIR} (separate volume from the real run)", flush=True)

    cmd = ["python3", "-m", "verl.trainer.main_ppo", *args]
    tail = collections.deque(maxlen=500)
    step_re = re.compile(r"training/global_step:(\d+)")
    last_step, t0 = 0, time.time()
    deadline = time.time() + (TRAINING_TIMEOUT_SECONDS - 300)
    timed_out = False

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            line = line.rstrip()
            tail.append(line)
            print(line, flush=True)
            m = step_re.search(line)
            if m:
                last_step = int(m.group(1))
            if time.time() > deadline:
                timed_out = True
                proc.terminate()
                break
        proc.wait(timeout=300)
    except Exception as e:  # noqa: BLE001 - report, never lose the checkpoint commit
        tail.append(f"EXCEPTION: {e}")
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        # Always commit: a crashed or timed-out run must still keep the
        # checkpoints it did produce (the lesson of commit 03b8c85).
        control_ckpt_volume.commit()

    return {
        "status": "timed_out" if timed_out else ("ok" if proc.returncode == 0 else "failed"),
        "returncode": proc.returncode,
        "last_step_seen": last_step,
        "elapsed_hr": round((time.time() - t0) / 3600, 2),
        "experiment_name": EXPERIMENT_NAME,
        "checkpoint_dir": CONTROL_CKPT_DIR,
        "tail": list(tail)[-40:],
    }


@app.function(
    image=image,
    gpu=EVAL_GPU,
    volumes={
        MODEL_CACHE_DIR: model_cache,
        CONTROL_CKPT_DIR: control_ckpt_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=8 * 60 * 60,
    secrets=[modal.Secret.from_dict({})],
)
def eval_control_model(which: str, n: int, problems: int, conditions: list[str], image_chunk: int) -> dict:
    """
    which: "base"      - untrained reference
           "real250"   - the real run's step-250 checkpoint (from the Hub)
           "random250" - this control's step-250 checkpoint (local volume)
    """
    import json
    import os
    import time

    os.environ["HF_HOME"] = MODEL_CACHE_DIR

    from src.data.gsm8k_loader import load_gsm8k
    from src.inference.checkpoint_loading import load_model_at_step
    from src.inference.run_sampling import run_sampling, save_sampling_records
    from src.metrics.pass_at_k import mean_pass_at_k

    print(f"=== control eval: {which} (n={n}, problems={problems}, {conditions}) ===", flush=True)
    t0 = time.time()

    kwargs = dict(model_id=MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    if which == "base":
        kwargs["step"] = None
    elif which == "real250":
        kwargs.update(step=CONTROL_STEP, repo_id=HF_CKPT_REPO, hf_token=os.environ.get("HF_TOKEN"))
    elif which == "random250":
        kwargs.update(step=CONTROL_STEP, local_checkpoint_dir=CONTROL_CKPT_DIR)
    else:
        raise ValueError(f"unknown which={which!r}")

    model, processor, vision_sha = load_model_at_step(**kwargs)

    examples = load_gsm8k("test")[:problems]
    records = run_sampling(
        model, processor, examples, conditions=conditions, n=n,
        progress_label=which, image_chunk=image_chunk,
    )

    out_path = f"{RESULTS_DIR}/control_records_{which}.parquet"
    save_sampling_records(records, out_path)
    results_volume.commit()

    per_cond = {}
    for cond in conditions:
        sub = [r for r in records if r.condition == cond]

        def _pp(attr: str):
            acc: dict[int, list[bool]] = {}
            for r in sub:
                acc.setdefault(r.problem_idx, []).append(getattr(r, attr))
            return [(len(v), sum(v)) for _, v in sorted(acc.items())]

        per_cond[cond] = {
            "pass_at_1": round(mean_pass_at_k(_pp("correct"), 1), 4),
            "pass_at_1_fallback": round(mean_pass_at_k(_pp("correct_fallback"), 1), 4),
            "format_compliance_rate": round(
                sum(1 for r in sub if r.extracted_answer is not None) / max(len(sub), 1), 4
            ),
        }

    summary = {
        "which": which, "n": n, "problems": len(examples), "conditions": conditions,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "vision_sha256": vision_sha, "per_condition": per_cond, "out_path": out_path,
    }
    with open(f"{RESULTS_DIR}/control_summary_{which}.json", "w") as f:
        json.dump(summary, f, indent=2)
    results_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=60 * 60)
def analyze_control(conditions: str = "T,E") -> dict:
    """
    The control's verdict: does the gain survive when the reward is
    severed from correctness?

    Reports Delta(real250 - base) and Delta(random250 - base) side by
    side with paired bootstrap CIs, and the RATIO between them - the
    fraction of the measured gain that a meaningless reward reproduces.
    """
    import json

    from src.inference.run_sampling import load_sampling_records
    from src.metrics.bootstrap_ci import bootstrap_delta_ci

    # Comma-separated string, not list[str] - see analyze_variant.
    conditions = [c.strip() for c in conditions.replace(",", " ").split()
                  if c.strip() in ("T", "D", "E")] or EVAL_CONDITIONS

    def per_problem(which: str, condition: str, col: str = "correct"):
        df = load_sampling_records(f"{RESULTS_DIR}/control_records_{which}.parquet")
        g = df[df["condition"] == condition].groupby("problem_idx")[col]
        return [(int(c), int(s)) for c, s in zip(g.count(), g.sum())]

    out = {"step": CONTROL_STEP, "per_condition": {}}
    for condition in conditions:
        # Scored format-agnostically. Under strict scoring BOTH runs would
        # show a "gain" that is largely format compliance - and the random
        # reward can teach formatting just as well as the real one, since
        # a Bernoulli(0.5) reward still rewards half of all well-formatted
        # answers. Only the fallback column isolates reasoning.
        col = "correct_fallback"
        base = per_problem("base", condition, col)
        real = bootstrap_delta_ci(per_problem("real250", condition, col), base, k=1)
        rand = bootstrap_delta_ci(per_problem("random250", condition, col), base, k=1)
        base_s = per_problem("base", condition)
        real_s = bootstrap_delta_ci(per_problem("real250", condition), base_s, k=1)
        rand_s = bootstrap_delta_ci(per_problem("random250", condition), base_s, k=1)

        d_real, d_rand = real["point_estimate"], rand["point_estimate"]
        ratio = (d_rand / d_real) if abs(d_real) > 1e-9 else None

        if not rand["significant"] and real["significant"]:
            verdict = (
                "PASSES: the real reward produces a significant gain at this step while the "
                "randomised reward does not. The spurious-reward explanation is refuted on "
                "this model and this task, not merely by analogy."
            )
        elif rand["significant"] and ratio is not None and ratio >= 0.5:
            verdict = (
                f"FAILS: the randomised reward reproduces {ratio:.0%} of the real gain. A large "
                f"part of the measured Delta is reward-independent and the headline claim must "
                f"be reframed accordingly - this is a finding, not a failed experiment."
            )
        elif rand["significant"]:
            verdict = (
                f"PARTIAL: the randomised reward produces a smaller but significant gain "
                f"({ratio:.0%} of the real gain). Report the real gain net of this, and state "
                f"the reward-independent component explicitly."
            )
        else:
            verdict = (
                "INCONCLUSIVE: neither delta is significant at this step and sample size. "
                "Check that the real-reward delta at step 250 is itself large enough to be "
                "detectable before drawing any conclusion from the control's null."
            )

        out["per_condition"][condition] = {
            "delta_real_minus_base": {
                "delta": round(d_real, 4),
                "ci": [round(real["ci_lower"], 4), round(real["ci_upper"], 4)],
                "significant": real["significant"],
            },
            "delta_random_minus_base": {
                "delta": round(d_rand, 4),
                "ci": [round(rand["ci_lower"], 4), round(rand["ci_upper"], 4)],
                "significant": rand["significant"],
            },
            "random_share_of_real_gain": round(ratio, 4) if ratio is not None else None,
            "strict_scoring_for_reference": {
                "delta_real": round(real_s["point_estimate"], 4),
                "delta_random": round(rand_s["point_estimate"], 4),
                "note": (
                    "Strict scoring conflates formatting with reasoning. If BOTH the real and "
                    "the random reward show a strict gain, that is direct evidence the gain is "
                    "format compliance - a meaningless reward cannot teach reasoning."
                ),
            },
            "verdict": verdict,
        }

    with open(f"{RESULTS_DIR}/control_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    results_volume.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def evaluate(n: int = EVAL_N, problems: int = EVAL_PROBLEMS, conditions: str = "TE",
             image_chunk: int = 32):
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "HF_TOKEN is required - the REAL run's step-250 checkpoint lives on the HF Hub."
        )
    cond_list = [c for c in conditions if c in ("T", "D", "E")]
    fn = eval_control_model.with_options(secrets=[modal.Secret.from_dict({"HF_TOKEN": hf_token})])

    print(f"Evaluating base / real250 / random250 at n={n}, problems={problems}, {cond_list}")
    calls = {}
    for which in ("base", "real250", "random250"):
        calls[which] = fn.spawn(which, n, problems, cond_list, image_chunk)
        print(f"  {which}: call_id={calls[which].object_id}")
    print("\nWhen finished:")
    print(f'  modal run scripts/run_random_reward_control_on_modal.py::analyze_control '
          f'--conditions "{",".join(cond_list)}"')
