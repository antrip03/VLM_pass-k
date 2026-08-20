"""
Re-score already-generated eval records with the current answer extractor,
in place, without re-running any generation.

Why this is safe and cheap: run_sampling stores the FULL completion text
alongside the extracted answer, so extraction is a pure function of data
already on disk. When src/metrics/answer_extraction.py changes - as it did
2026-08-20, to tolerate "#### <18>" / "#### \\$18" decoration that was
scoring ~21-27% of real completions as no-answer - the fix can be applied
to existing 19,200-record runs for free, instead of paying ~4 GPU-hours
per model to regenerate identical text.

Reports before/after so the effect of an extractor change on the headline
numbers is explicit and auditable, never silent.

Usage:
    modal run scripts/rescore_records.py --model-kind base
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal

from configs.modal_app_verl import app, image

RESULTS_DIR = "/results"
results_volume = modal.Volume.from_name("grpo-vlm-phase1-eval-results", create_if_missing=True)


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=30 * 60)
def rescore(model_kind: str) -> dict:
    import pandas as pd

    from src.metrics.answer_extraction import extract_model_answer

    path = f"{RESULTS_DIR}/records_{model_kind}.parquet"
    df = pd.read_parquet(path)

    before = {
        c: {
            "accuracy": round(float(df[df.condition == c]["correct"].mean()), 4),
            "no_answer_rate": round(float(df[df.condition == c]["extracted_answer"].isna().mean()), 4),
        }
        for c in ["T", "D", "E"]
    }

    df["extracted_answer"] = df["completion"].map(extract_model_answer)
    df["correct"] = df["extracted_answer"].notna() & (df["extracted_answer"] == df["ground_truth"])

    after = {
        c: {
            "accuracy": round(float(df[df.condition == c]["correct"].mean()), 4),
            "no_answer_rate": round(float(df[df.condition == c]["extracted_answer"].isna().mean()), 4),
        }
        for c in ["T", "D", "E"]
    }

    # Keep the original file recoverable rather than overwriting blind.
    backup = f"{RESULTS_DIR}/records_{model_kind}.prefix_extractor.parquet"
    if not Path(backup).exists():
        pd.read_parquet(path).to_parquet(backup)
    df.to_parquet(path)
    results_volume.commit()

    return {"model_kind": model_kind, "records": len(df), "before": before, "after": after,
            "backup": backup}


@app.local_entrypoint()
def main(model_kind: str = "base"):
    import json

    print(json.dumps(rescore.remote(model_kind), indent=2))
