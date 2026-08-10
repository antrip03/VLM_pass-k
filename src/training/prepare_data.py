"""
Converts GSM8K into veRL's expected Parquet schema (PLAN.md Section 3
Step 5). Schema confirmed against TWO real, independent sources (not
guessed): veRL's own examples/data_preprocess/gsm8k.py (fetched
verbatim, 2026-08-10) and a complete, working Modal+veRL+GRPO example
(modal.com/docs/examples/grpo_verl, also fetched verbatim same day) -
both produce/expect exactly:
  data_source: str
  prompt: list[{"role": "user", "content": str}]
  ability: str
  reward_model: {"style": "rule", "ground_truth": str}
  extra_info: dict (free-form; split/index/original text kept here)

Deliberately does NOT reuse veRL's own gsm8k.py script as-is, despite it
existing and being schema-correct - it appends a weaker instruction
("Let's think step by step and output the final answer after \"####\".")
than this project's own validated src/data/prompts.build_text_prompt(),
which was specifically strengthened after a live bug: the base model
frequently reasoned correctly but wrote "\\boxed{N}" instead of "####",
because the original soft instruction didn't reliably override that
pretrained habit (see src/data/prompts.py's docstring and
src/metrics/answer_extraction.py's \\boxed{} fallback - complementary
fixes, not redundant). Using the same prompt template for training data
as for evaluation also keeps the two consistent, which matters for
interpreting Delta_text/Delta_pixel cleanly.

Ground truth extraction reuses src/data/gsm8k_loader.extract_final_answer
directly - already regex-compatible with veRL's own extraction (both use
"#### (-?[0-9.,]+)"), confirmed by inspection of the real veRL source,
not assumed.

Usage (local, no GPU needed):
    python -m src.training.prepare_data --split train --out data/gsm8k_train.parquet
    python -m src.training.prepare_data --split test --out data/gsm8k_test.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.gsm8k_loader import load_gsm8k
from src.data.prompts import build_text_prompt

DATA_SOURCE = "openai/gsm8k"
ABILITY = "math"


def build_verl_rows(split: str) -> list[dict]:
    """
    Returns one dict per GSM8K example, matching veRL's real Parquet
    schema exactly. Kept as a plain list-of-dicts (not written to disk)
    so this is unit-testable without touching the filesystem.
    """
    examples = load_gsm8k(split)
    rows = []
    for ex in examples:
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": build_text_prompt(ex.question)}],
                "ability": ABILITY,
                "reward_model": {"style": "rule", "ground_truth": ex.final_answer},
                "extra_info": {
                    "split": ex.split,
                    "index": ex.idx,
                    "raw_answer": ex.answer_text,
                    "question": ex.question,
                },
            }
        )
    return rows


def write_parquet(split: str, out_path: str | Path) -> Path:
    rows = build_verl_rows(split)
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    written = write_parquet(args.split, args.out)
    print(f"Wrote {written}")
