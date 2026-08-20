"""
Diagnose the no-extractable-answer rate found in the Phase 1 base eval
(T 21.2%, D 21.5%, E 26.9%) - PLAN.md Phase 1 item 7's mandatory
truncation check, plus item 5's extraction spot-check, on real data.

These two causes look identical in the summary but mean opposite things:

  TRUNCATION - the generation hit the 1200-token cap before reaching an
  answer. The sample is genuinely unfinished; the model may or may not
  have been on track. Fix is to raise the cap and re-run. PLAN.md flags
  ASYMMETRY here (one condition or model truncating more) as the
  dangerous case, since it biases the comparison.

  EXTRACTION FAILURE - the model did answer, but not in a form
  extract_model_answer() recognises (it expects #### or \\boxed{}). The
  sample is fine and is being scored as wrong purely by regex. This
  inflates the apparent gap between conditions if image-mode phrasing
  habits differ from text-mode ones - exactly the risk PLAN.md item 5
  warns about ("a regex tuned on text-mode output isn't guaranteed to
  match image-mode phrasing").

Distinguishing them: a truncated completion is long (near the cap in
tokens) and typically ends mid-sentence; an extraction failure is short,
terminates naturally, and usually contains a number near the end. This
reports the length distribution of no-answer completions per condition
and prints real examples for eyeballing.

Usage:
    modal run scripts/diagnose_no_answer.py
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
def diagnose(model_kind: str, examples_per_condition: int) -> dict:
    import re

    from src.inference.run_sampling import load_sampling_records

    df = load_sampling_records(f"{RESULTS_DIR}/records_{model_kind}.parquet")
    out = {"model_kind": model_kind, "conditions": {}}

    for condition in ["T", "D", "E"]:
        sub = df[df["condition"] == condition]
        no_ans = sub[sub["extracted_answer"].isna()]
        has_ans = sub[sub["extracted_answer"].notna()]

        # Character length is a good enough proxy for "did this run to the
        # cap" - the 1200-token cap is roughly 4000+ characters.
        lens = no_ans["completion"].str.len()
        # Does the unparsed completion still end with a number? Strong hint
        # that the answer IS there and only the FORMAT was missed.
        ends_with_number = no_ans["completion"].str.strip().str.endswith(
            tuple(str(d) for d in range(10))
        )
        has_digit_in_tail = no_ans["completion"].str[-120:].str.contains(r"\d", regex=True, na=False)

        out["conditions"][condition] = {
            "n_total": int(len(sub)),
            "n_no_answer": int(len(no_ans)),
            "no_answer_rate": round(len(no_ans) / max(len(sub), 1), 4),
            "no_answer_len_chars": {
                "mean": int(lens.mean()) if len(lens) else 0,
                "median": int(lens.median()) if len(lens) else 0,
                "p90": int(lens.quantile(0.9)) if len(lens) else 0,
                "max": int(lens.max()) if len(lens) else 0,
            },
            "answered_len_chars_mean": int(has_ans["completion"].str.len().mean()) if len(has_ans) else 0,
            "no_answer_ends_with_digit_rate": round(float(ends_with_number.mean()), 4) if len(no_ans) else 0.0,
            "no_answer_digit_in_last_120_chars": round(float(has_digit_in_tail.mean()), 4) if len(no_ans) else 0.0,
            "examples": [
                {"tail": c[-400:], "len": len(c)}
                for c in no_ans["completion"].head(examples_per_condition).tolist()
            ],
        }
    return out


@app.local_entrypoint()
def main(model_kind: str = "base", examples_per_condition: int = 3):
    import json

    res = diagnose.remote(model_kind, examples_per_condition)
    for cond, d in res["conditions"].items():
        ex = d.pop("examples")
        print(f"\n=== condition {cond} ===")
        print(json.dumps(d, indent=2))
        for i, e in enumerate(ex, 1):
            print(f"\n--- {cond} no-answer example {i} (len={e['len']} chars), TAIL: ---")
            print(e["tail"])
