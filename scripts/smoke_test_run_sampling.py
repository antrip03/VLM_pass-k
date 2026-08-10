"""
Real, cheap smoke test for src/inference/run_sampling.py - the
orchestration layer built on top of the already-validated sampler
functions (scripts/smoke_test_sampler.py). Small scale (2 problems x 3
conditions x n=3 = 18 total generations), but exercises the FULL Step 6
pipeline: sampling -> record construction -> Parquet save -> reload,
using a real Modal Volume for persistence (not just local disk, since
the real pipeline needs to survive across separate Modal function calls).

Usage: modal run scripts/smoke_test_run_sampling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app import MODEL_CACHE_DIR, MODEL_ID, app, image, model_cache

RESULTS_DIR = "/results"
results_volume = modal.Volume.from_name("grpo-vlm-sampling-results", create_if_missing=True)

N_PROBLEMS = 2
N_SAMPLES = 3


@app.function(
    image=image,
    gpu="A10G",
    volumes={MODEL_CACHE_DIR: model_cache, RESULTS_DIR: results_volume},
    timeout=15 * 60,
)
def smoke_test_run_sampling() -> dict:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.data.gsm8k_loader import load_gsm8k
    from src.inference.run_sampling import load_sampling_records, run_sampling, save_sampling_records

    result = {"steps": []}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=MODEL_CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    result["steps"].append("Model + processor loaded")

    problems = load_gsm8k("test")[:N_PROBLEMS]
    records = run_sampling(model, processor, problems, conditions=["T", "D", "E"], n=N_SAMPLES)
    result["steps"].append(f"run_sampling produced {len(records)} records")
    expected = N_PROBLEMS * 3 * N_SAMPLES
    result["expected_record_count"] = expected
    result["actual_record_count"] = len(records)

    out_path = f"{RESULTS_DIR}/smoke_test.parquet"
    save_sampling_records(records, out_path)
    results_volume.commit()
    result["steps"].append(f"Saved to {out_path}")

    reloaded = load_sampling_records(out_path)
    result["steps"].append(f"Reloaded {len(reloaded)} rows from Parquet")
    result["reloaded_row_count"] = len(reloaded)
    result["reloaded_columns"] = list(reloaded.columns)

    per_condition = reloaded.groupby("condition").agg(
        n=("correct", "size"), n_correct=("correct", "sum"), n_extracted=("extracted_answer", lambda s: s.notna().sum())
    )
    result["per_condition_summary"] = per_condition.to_dict(orient="index")

    result["status"] = "PASS" if len(records) == expected and len(reloaded) == expected else "FAIL"
    return result


@app.local_entrypoint()
def run():
    result = smoke_test_run_sampling.remote()
    for step in result["steps"]:
        print(f"  - {step}")
    print(f"\nExpected records: {result['expected_record_count']}, actual: {result['actual_record_count']}")
    print(f"Reloaded rows: {result['reloaded_row_count']}")
    print(f"Columns: {result['reloaded_columns']}")
    print("\nPer-condition summary:")
    for cond, stats in result["per_condition_summary"].items():
        print(f"  {cond}: {stats}")
    print(f"\nStatus: {result['status']}")
