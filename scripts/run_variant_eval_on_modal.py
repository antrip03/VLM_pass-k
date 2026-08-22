"""
Phase 1b Steps 2, 3 and 6: evaluate the Phase 1 checkpoints on a
PERTURBED variant of the same 50 problems.

ONE SCRIPT, THREE JOBS - deliberately, because they are the same
computation with different data:

  Step 2  --dataset gsmplus_harder --mode screen
          Does the harder set actually break the pass@64 ceiling? Base
          model only, ~20 problems, Condition T. Costs well under $1 and
          decides whether Step 3 is worth running at all. Do not skip it:
          if the harder set still saturates, Step 3 buys nothing.

  Step 3  --dataset gsmplus_harder --mode full
          The ceiling fix. Both checkpoints, all conditions, n=128. This
          is the single highest-value experiment in the plan, because it
          is the only one that can turn Finding 2 (sharpening) from a
          measurement-at-saturation into a real test.

  Step 6  --dataset gsmplus_rephrased --mode full --conditions T
          The paraphrase-OOD control, replacing the name-swap stub in
          src/data/paraphrase_ood.py.

WHY THE SAME 50 PROBLEMS
------------------------
Verified 2026-08-22: every one of this project's 50 evaluation problems
has a GSM-Plus variant, for both the rephrase and the harder
perturbations. Evaluating variants of the SAME problems - rather than a
fresh sample - holds problem selection constant, so any accuracy change
is attributable to the perturbation itself. It also preserves problem_idx,
so these records pair directly against the existing Phase 1 parquets for
the paired bootstrap.

WHAT THE REPHRASE CONTROL DOES NOT NEED
---------------------------------------
It does not need the originals re-run: Phase 1 already measured them on
exactly these problems. Only the rephrased variants are generated here,
and the comparison is made against the existing records.

    modal run scripts/run_variant_eval_on_modal.py --dataset gsmplus_harder --mode screen
    modal run scripts/run_variant_eval_on_modal.py --dataset gsmplus_harder --mode full
    modal run scripts/run_variant_eval_on_modal.py --dataset gsmplus_rephrased --mode full --conditions T
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
FINAL_STEP = 467
EVAL_GPU = "A10G"

# Screen mode is intentionally tiny: its only job is to answer "does base
# pass@64 come off the ceiling on this set?", which needs a rough number,
# not a publishable one.
SCREEN_PROBLEMS = 20
SCREEN_N = 64


@app.function(
    image=image,
    gpu=EVAL_GPU,
    volumes={MODEL_CACHE_DIR: model_cache, RESULTS_DIR: results_volume},
    timeout=20 * 60 * 60,
    secrets=[modal.Secret.from_dict({})],
)
def eval_variant(
    dataset: str,
    model_kind: str,
    n: int,
    problems: int,
    conditions: list[str],
    image_chunk: int,
    tag: str,
) -> dict:
    """
    Evaluate one model on one perturbed variant set.

    model_kind: "base" (untrained) or "rl" (the step-467 checkpoint).
    """
    import json
    import os
    import time

    os.environ["HF_HOME"] = MODEL_CACHE_DIR

    from src.data.gsm8k_loader import load_gsm8k
    from src.data.harder_datasets import HARDER_DATASETS
    from src.inference.checkpoint_loading import load_model_at_step
    from src.inference.run_sampling import run_sampling, save_sampling_records
    from src.metrics.pass_at_k import mean_pass_at_k

    print(f"=== eval_variant(dataset={dataset}, model={model_kind}, n={n}, "
          f"problems={problems}, conditions={conditions}) ===", flush=True)
    t0 = time.time()

    # Always derive the variant from THIS project's canonical eval slice,
    # so problem_idx lines up with the Phase 1 records.
    base_problems = load_gsm8k("test")[:problems]
    loader = HARDER_DATASETS[dataset]
    examples, ds_stats = loader(base_problems)
    print(json.dumps({k: v for k, v in ds_stats.items()
                      if k != "chosen_type_by_problem_idx"}, indent=2), flush=True)
    if ds_stats["problems_matched"] != len(base_problems):
        print(f"WARNING: only {ds_stats['problems_matched']}/{len(base_problems)} problems had a "
              f"variant; missing idx={ds_stats['problems_missing']}", flush=True)

    model, processor, vision_sha = load_model_at_step(
        model_id=MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        step=None if model_kind == "base" else FINAL_STEP,
        repo_id=HF_CKPT_REPO,
        hf_token=os.environ.get("HF_TOKEN"),
    )

    records = run_sampling(
        model, processor, examples, conditions=conditions, n=n,
        progress_label=f"{tag}/{model_kind}", image_chunk=image_chunk,
    )

    out_path = f"{RESULTS_DIR}/{tag}_records_{model_kind}.parquet"
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

        pp, pp_fb = _pp("correct"), _pp("correct_fallback")
        per_cond[cond] = {
            # Strict = what training optimised. Fallback = format-agnostic.
            # Both are reported because the DIFFERENCE is the finding.
            "pass_at_1": round(mean_pass_at_k(pp, 1), 4),
            "pass_at_1_fallback": round(mean_pass_at_k(pp_fb, 1), 4),
            "pass_at_8": round(mean_pass_at_k(pp, 8), 4) if n >= 8 else None,
            "pass_at_64": round(mean_pass_at_k(pp, 64), 4) if n >= 64 else None,
            "pass_at_64_fallback": round(mean_pass_at_k(pp_fb, 64), 4) if n >= 64 else None,
            "no_answer_rate": round(
                sum(1 for r in sub if r.extracted_answer is None) / max(len(sub), 1), 4
            ),
            "format_compliance_rate": round(
                sum(1 for r in sub if r.extracted_answer is not None) / max(len(sub), 1), 4
            ),
        }

    summary = {
        "tag": tag,
        "dataset": dataset,
        "model_kind": model_kind,
        "n": n,
        "problems": len(examples),
        "conditions": conditions,
        "records": len(records),
        "elapsed_hr": round((time.time() - t0) / 3600, 2),
        "vision_sha256": vision_sha,
        "dataset_stats": {k: v for k, v in ds_stats.items() if k != "chosen_type_by_problem_idx"},
        "per_condition": per_cond,
        "out_path": out_path,
    }
    with open(f"{RESULTS_DIR}/{tag}_summary_{model_kind}.json", "w") as f:
        json.dump(summary, f, indent=2)
    results_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=60 * 60)
def analyze_variant(tag: str, conditions: str = "T,D,E") -> dict:
    """
    pass@k and paired-bootstrap Deltas on a variant set, plus the
    ceiling verdict that decides whether Finding 2 is testable here.
    """
    import json

    # Comma-separated string, not list[str]: invoked from the CLI,
    # where Modal passes the raw string through.
    conditions = [c.strip() for c in conditions.replace(",", " ").split() if c.strip() in ("T", "D", "E")]
    if not conditions:
        raise ValueError("conditions must name at least one of T, D, E")

    from src.analysis.coverage import full_coverage_report
    from src.inference.run_sampling import load_sampling_records
    from src.metrics.bootstrap_ci import bootstrap_delta_ci, bootstrap_pass_at_k_ci

    dfs = {}
    for kind in ("base", "rl"):
        df = load_sampling_records(f"{RESULTS_DIR}/{tag}_records_{kind}.parquet")
        df["model_kind"] = kind
        dfs[kind] = df

    def per_problem(df, condition, col="correct"):
        g = df[df["condition"] == condition].groupby("problem_idx")[col]
        return [(int(c), int(s)) for c, s in zip(g.count(), g.sum())]

    # Every quantity is computed under BOTH scoring rules. "strict" is the
    # rule training optimised (and which conflates formatting with
    # reasoning); "fallback" is format-agnostic. Reporting only one would
    # reproduce the Phase 1 error in either direction.
    out = {"tag": tag, "pass_at_k": {}, "deltas": {}}
    for scoring, col in (("strict", "correct"), ("fallback", "correct_fallback")):
        if col not in dfs["base"].columns:
            out.setdefault("warnings", []).append(
                f"column {col!r} absent - records predate dual scoring; "
                f"run scripts/rescore_with_fallback.py to add it"
            )
            continue
        for condition in conditions:
            for k in (1, 8, 64):
                for kind in ("base", "rl"):
                    res = bootstrap_pass_at_k_ci(per_problem(dfs[kind], condition, col), k=k)
                    out["pass_at_k"][f"{scoring}/{kind}/{condition}/k={k}"] = {
                        "mean": round(res["point_estimate"], 4),
                        "ci": [round(res["ci_lower"], 4), round(res["ci_upper"], 4)],
                    }
                d = bootstrap_delta_ci(
                    per_problem(dfs["rl"], condition, col),
                    per_problem(dfs["base"], condition, col),
                    k=k,
                )
                out["deltas"][f"{scoring}/Delta_{condition}/k={k}"] = {
                    "delta": round(d["point_estimate"], 4),
                    "ci": [round(d["ci_lower"], 4), round(d["ci_upper"], 4)],
                    "significant": d["significant"],
                }

    # THE question this whole step exists to answer.
    import pandas as pd

    combined = pd.concat([dfs["base"], dfs["rl"]], ignore_index=True)
    out["coverage"] = full_coverage_report(combined, conditions=tuple(conditions))

    verdicts = {}
    for condition in conditions:
        key = f"fallback/base/{condition}/k=64"
        if key not in out["pass_at_k"]:
            key = f"strict/base/{condition}/k=64"
        base64 = out["pass_at_k"][key]["mean"]
        if base64 >= 0.95:
            verdicts[condition] = (
                f"STILL SATURATED: base pass@64 = {base64:.4f}. This set does not fix the "
                f"ceiling - a flat Delta at k=64 here remains uninformative, exactly as on "
                f"plain GSM8K. Try a harder set or more aggressive perturbations."
            )
        elif base64 >= 0.85:
            verdicts[condition] = (
                f"PARTIAL HEADROOM: base pass@64 = {base64:.4f}. Some room to detect coverage "
                f"change, but a null result should still be reported as weak evidence."
            )
        else:
            verdicts[condition] = (
                f"CEILING BROKEN: base pass@64 = {base64:.4f}. There is now real dynamic range, "
                f"so a flat-or-declining pass@64 under RL is a genuine finding rather than an "
                f"artifact of saturation."
            )
    out["ceiling_verdict"] = verdicts

    with open(f"{RESULTS_DIR}/{tag}_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    results_volume.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.local_entrypoint()
def main(
    dataset: str = "gsmplus_harder",
    mode: str = "screen",
    conditions: str = "",
    n: int = 128,
    problems: int = 50,
    image_chunk: int = 32,
    tag: str = "",
):
    import os

    from src.data.harder_datasets import HARDER_DATASETS

    if dataset not in HARDER_DATASETS:
        raise SystemExit(f"--dataset must be one of {sorted(HARDER_DATASETS)}")
    if mode not in ("screen", "full"):
        raise SystemExit("--mode must be 'screen' or 'full'")

    hf_token = os.environ.get("HF_TOKEN")
    if mode == "full" and not hf_token:
        raise SystemExit(
            "HF_TOKEN is required for --mode full (the RL checkpoint lives on the HF Hub).\n"
            "  echo 'HF_TOKEN=hf_...' > .env && set -a && source .env && set +a"
        )

    if mode == "screen":
        # Base model only: the screen asks a question about the DATASET
        # (is it hard enough?), not about RL.
        kinds, n, problems = ["base"], SCREEN_N, SCREEN_PROBLEMS
        cond_list = ["T"]
    else:
        kinds = ["base", "rl"]
        cond_list = list(conditions) if conditions else ["T", "D", "E"]
        cond_list = [c for c in cond_list if c in ("T", "D", "E")]
        if not cond_list:
            raise SystemExit("--conditions must be a subset of T,D,E, e.g. --conditions T or TDE")

    tag = tag or f"{dataset}_{mode}"
    fn = eval_variant.with_options(
        secrets=[modal.Secret.from_dict({"HF_TOKEN": hf_token or ""})]
    )

    print(f"dataset={dataset} mode={mode} tag={tag}")
    print(f"models={kinds} conditions={cond_list} n={n} problems={problems}")
    if mode == "screen":
        print("Screen is base-only, T-only, 20 problems x n=64 - expected well under $1.")
    else:
        print("FULL RUN - this is real spend. Confirm you intended it before leaving it running.")

    calls = {}
    for kind in kinds:
        calls[kind] = fn.spawn(dataset, kind, n, problems, cond_list, image_chunk, tag)
        print(f"  {kind}: call_id={calls[kind].object_id}")

    print("\nSpawned - safe to disconnect. When finished:")
    if mode == "full":
        print(f'  modal run scripts/run_variant_eval_on_modal.py::analyze_variant '
              f'--tag "{tag}" --conditions "{",".join(cond_list)}"')
    else:
        print(f"  modal volume get grpo-vlm-phase1b-results {tag}_summary_base.json .")
        print("  Look at per_condition.T.pass_at_64 - if it is still >= 0.95, this set does not "
              "fix the ceiling and Step 3 is not worth running on it.")
