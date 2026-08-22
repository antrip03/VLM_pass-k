"""
Phase 1b Step 1: Delta_text as a function of TRAINING STEP.

WHAT QUESTION THIS ANSWERS, AND WHAT IT DOES NOT
------------------------------------------------
Shao et al. (arXiv 2506.10947) report that spurious reward signals
produce gains that appear "within the first 50 steps" and then plateau,
whereas a signal genuinely tied to correctness keeps paying off with more
training. So the SHAPE of Delta_text over training carries evidence about
which of those we have, and it costs almost nothing because all 18
checkpoints already exist (save_freq=25 over 467 steps).

Be precise about the strength of that evidence, though, because it is
narrower than it first appears:

  * A curve that is FLAT after ~step 50 is a genuine warning sign. It
    matches the fast-plateau signature of the format / incorrect-label
    spurious rewards.

  * A curve that CLIMBS steadily does NOT by itself rule out a spurious
    explanation. Shao et al. specifically note the RANDOM reward
    "converges slower than the other spurious rewards", needing >100
    steps on smaller models. A slow climb is consistent with both real
    learning and random-reward dynamics.

So this is a prior-shifter and a cheap early warning, NOT a substitute
for the randomised-reward control (Step 4). It is run first because it is
nearly free and because a flat curve would change what Step 4 is worth
spending money on.

Note also that the format-compliance component of the reward (the
confound fixed in commit 43279ab) is precisely a fast-plateau mechanism -
the model can learn to emit "#### N" cleanly in very few steps. Since
that confound has already been corrected in the SCORING, an early plateau
in this curve would now point at something else.

CONDITION T ONLY - deliberately. The quantity under challenge is
Delta_text; D and E cost far more per checkpoint (image conditions) and
add nothing to the shape question.

    modal run scripts/run_trajectory_eval_on_modal.py
    modal run scripts/run_trajectory_eval_on_modal.py --steps 25,100,200,467 --n 128

COST: measured T throughput is ~27.9 s/problem, i.e. ~23 min per
checkpoint at 50 problems x n=128. The default 5-point sweep
(base + 4 steps) is therefore ~2hr wall-clock across parallel
containers, ~$2-3 at the rate the Phase 1 eval actually billed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

RESULTS_DIR = "/results"
results_volume = modal.Volume.from_name("grpo-vlm-phase1b-results", create_if_missing=True)

HF_CKPT_REPO = "GunGG4/grpo-vlm-phase1-checkpoints"
EVAL_GPU = "A10G"

# Default sweep. 25 is the earliest checkpoint (the fast-plateau window
# Shao et al. describe); 467 is the final model whose Delta_text=+0.0587
# is the number being defended. Every value must be a real saved
# checkpoint: save_freq=25, so multiples of 25 up to 450, plus 467.
DEFAULT_STEPS = (25, 100, 200, 467)


def _valid_step(step: int) -> bool:
    return step == 467 or (step % 25 == 0 and 25 <= step <= 450)


@app.function(
    image=image,
    gpu=EVAL_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, RESULTS_DIR: results_volume},
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_dict({})],  # replaced per-call with the real HF token
)
def eval_checkpoint_text_only(step: int | None, n: int, problems: int) -> dict:
    """
    Condition-T pass@1 (and pass@k) for one checkpoint.

    step=None evaluates the untrained base model, giving the trajectory a
    self-contained zero point rather than depending on records that live
    on a different Modal account.
    """
    import json
    import os
    import time

    os.environ["HF_HOME"] = MODEL_CACHE_DIR

    from src.data.gsm8k_loader import load_gsm8k
    from src.inference.checkpoint_loading import load_model_at_step
    from src.inference.run_sampling import run_sampling, save_sampling_records
    from src.metrics.pass_at_k import mean_pass_at_k

    label = "base" if step is None else f"step_{step}"
    print(f"=== trajectory eval: {label} (n={n}, problems={problems}) ===", flush=True)
    t0 = time.time()

    model, processor, vision_sha = load_model_at_step(
        model_id=MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        step=step,
        repo_id=HF_CKPT_REPO,
        hf_token=os.environ.get("HF_TOKEN"),
    )

    examples = load_gsm8k("test")[:problems]
    records = run_sampling(model, processor, examples, conditions=["T"], n=n, progress_label=label)

    out_path = f"{RESULTS_DIR}/trajectory_records_{label}.parquet"
    save_sampling_records(records, out_path)
    results_volume.commit()

    # (n, c) per problem, ordered by problem_idx so every checkpoint's
    # list is aligned with every other's for later paired comparisons.
    def _per_problem(attr: str):
        acc: dict[int, list[bool]] = {}
        for r in records:
            acc.setdefault(r.problem_idx, []).append(getattr(r, attr))
        return [(len(v), sum(v)) for _, v in sorted(acc.items())]

    per_problem = _per_problem("correct")
    per_problem_fb = _per_problem("correct_fallback")

    # FORMAT COMPLIANCE is now a headline quantity, not a diagnostic: the
    # Phase 1 finding is that RL's apparent gain WAS this. Tracking it per
    # checkpoint turns the mechanism into a directly plottable curve
    # against the reasoning curve.
    compliance = sum(1 for r in records if r.extracted_answer is not None) / max(len(records), 1)

    summary = {
        "label": label,
        "step": step,
        "n": n,
        "problems": len(examples),
        "records": len(records),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "vision_sha256": vision_sha,
        "format_compliance_rate": round(compliance, 4),
        "pass_at_1": round(mean_pass_at_k(per_problem, 1), 4),
        "pass_at_1_fallback": round(mean_pass_at_k(per_problem_fb, 1), 4),
        "pass_at_8": round(mean_pass_at_k(per_problem, 8), 4) if n >= 8 else None,
        "pass_at_64": round(mean_pass_at_k(per_problem, 64), 4) if n >= 64 else None,
        "pass_at_64_fallback": round(mean_pass_at_k(per_problem_fb, 64), 4) if n >= 64 else None,
        "no_answer_rate": round(
            sum(1 for r in records if r.extracted_answer is None) / max(len(records), 1), 4
        ),
        "out_path": out_path,
    }
    with open(f"{RESULTS_DIR}/trajectory_summary_{label}.json", "w") as f:
        json.dump(summary, f, indent=2)
    results_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=60 * 60)
def analyze_trajectory(labels: str) -> dict:
    """
    Assemble the per-checkpoint summaries into a trajectory, with paired
    bootstrap CIs on Delta_text at each step relative to base.

    Reports the shape verdict explicitly rather than leaving it to the
    reader, and states the interpretive limit in the same breath so the
    number is never quoted without it.
    """
    import json

    from src.inference.run_sampling import load_sampling_records
    from src.metrics.bootstrap_ci import bootstrap_delta_ci

    # Accept a comma-separated string: `modal run ...::analyze_trajectory
    # --labels "base,step_25"` passes a STRING, not a list, so a
    # list[str] annotation would receive one long label and fail on a
    # missing file rather than on the real cause.
    labels = [x.strip() for x in labels.split(",") if x.strip()]
    if "base" not in labels:
        raise ValueError("labels must include 'base' - it is the reference for every Delta")

    def per_problem(label: str, col: str = "correct"):
        df = load_sampling_records(f"{RESULTS_DIR}/trajectory_records_{label}.parquet")
        g = df[df["condition"] == "T"].groupby("problem_idx")[col]
        return [(int(c), int(s)) for c, s in zip(g.count(), g.sum())]

    summaries = {}
    for label in labels:
        with open(f"{RESULTS_DIR}/trajectory_summary_{label}.json") as f:
            summaries[label] = json.load(f)

    base_pp = per_problem("base")
    base_pp_fb = per_problem("base", "correct_fallback")
    trajectory = []
    for label in labels:
        if label == "base":
            continue
        d = bootstrap_delta_ci(per_problem(label), base_pp, k=1)
        d_fb = bootstrap_delta_ci(per_problem(label, "correct_fallback"), base_pp_fb, k=1)
        trajectory.append(
            {
                "label": label,
                "step": summaries[label]["step"],
                # The two curves whose DIVERGENCE is the mechanism: format
                # compliance should climb steeply while format-agnostic
                # reasoning stays flat, if the Phase 1 diagnosis is right.
                "format_compliance_rate": summaries[label].get("format_compliance_rate"),
                "pass_at_1_strict": summaries[label]["pass_at_1"],
                "pass_at_1_fallback": summaries[label].get("pass_at_1_fallback"),
                "delta_text_strict": round(d["point_estimate"], 4),
                "ci_strict": [round(d["ci_lower"], 4), round(d["ci_upper"], 4)],
                "significant_strict": d["significant"],
                "delta_text_fallback": round(d_fb["point_estimate"], 4),
                "ci_fallback": [round(d_fb["ci_lower"], 4), round(d_fb["ci_upper"], 4)],
                "significant_fallback": d_fb["significant"],
            }
        )
    trajectory.sort(key=lambda r: r["step"])

    verdict = None
    if len(trajectory) >= 2:
        first, last = trajectory[0], trajectory[-1]
        growth = last["delta_text_fallback"] - first["delta_text_fallback"]
        # "Plateau" is defined against the CI width rather than an
        # arbitrary absolute threshold: growth smaller than the noise on
        # a single point is not growth.
        noise = (first["ci_fallback"][1] - first["ci_fallback"][0]) / 2
        if growth <= noise:
            verdict = (
                f"FLAT: Delta_text at step {last['step']} ({last['delta_text']}) is within "
                f"noise of step {first['step']} ({first['delta_text']}). This matches the "
                f"fast-plateau signature of spurious rewards and is a warning sign - the "
                f"randomised-reward control becomes essential, not optional."
            )
        else:
            verdict = (
                f"CLIMBING: Delta_text grows {growth:+.4f} from step {first['step']} to "
                f"{last['step']}, beyond single-point noise ({noise:.4f}). Consistent with a "
                f"reward signal carrying real information - but NOT conclusive: Shao et al. "
                f"note random rewards also converge slowly (>100 steps on small models), so "
                f"this does not replace the randomised-reward control."
            )

    out = {"base_pass_at_1_strict": summaries["base"]["pass_at_1"],
           "base_pass_at_1_fallback": summaries["base"].get("pass_at_1_fallback"), "trajectory": trajectory,
           "verdict": verdict}
    with open(f"{RESULTS_DIR}/trajectory_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    results_volume.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(steps: str = "", n: int = 128, problems: int = 50, include_base: bool = True):
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "HF_TOKEN is required (checkpoints live on the HF Hub mirror).\n"
            "  echo 'HF_TOKEN=hf_...' > .env   # .env is git-ignored\n"
            "  set -a; source .env; set +a"
        )

    step_list = [int(s) for s in steps.split(",") if s.strip()] if steps else list(DEFAULT_STEPS)
    bad = [s for s in step_list if not _valid_step(s)]
    if bad:
        raise SystemExit(
            f"Steps {bad} are not saved checkpoints. save_freq=25 over 467 total steps, so "
            f"valid values are multiples of 25 from 25 to 450, plus 467."
        )

    fn = eval_checkpoint_text_only.with_options(
        secrets=[modal.Secret.from_dict({"HF_TOKEN": hf_token})]
    )

    targets: list[int | None] = ([None] if include_base else []) + step_list
    print(f"Spawning {len(targets)} Condition-T evals in parallel on {EVAL_GPU} "
          f"(n={n}, problems={problems})")
    print(f"Estimated: ~23 min each at n=128/50 problems; they run concurrently.")

    calls = {}
    for step in targets:
        label = "base" if step is None else f"step_{step}"
        calls[label] = fn.spawn(step, n, problems)
        print(f"  {label}: call_id={calls[label].object_id}")

    print("\nAll spawned - safe to disconnect. When they finish, run:")
    labels = ",".join(calls)
    print(f'  modal run scripts/run_trajectory_eval_on_modal.py::analyze_trajectory --labels "{labels}"')
