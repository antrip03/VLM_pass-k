"""
Sanity-check script for the Step 2 data pipeline (PLAN.md Section 3).

Runs a small, fast, local (no GPU/Modal needed) end-to-end check:
  - loads a small sample of GSM8K train + test
  - verifies answer extraction against known values
  - renders a handful of images and checks basic sanity (non-trivial size,
    no exceptions)
  - runs the leakage check and the text/image pairing-integrity check
  - runs the paraphrase stub and confirms numbers are preserved

This is a debug/validation script, not part of the production pipeline -
exactly the kind of small, cheap check meant to run before spending real
compute on later steps (PLAN.md's phased build-out).

Usage: python scripts/sanity_check_data.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k_loader import load_gsm8k, extract_final_answer
from src.data.leakage_check import check_train_eval_leakage, check_text_image_pairing
from src.data.paraphrase_ood import paraphrase_stub
from src.data.render import render_problem_image

OUT_DIR = Path(__file__).resolve().parent.parent / "scratch" / "sanity_check_images"

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _all_numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def main() -> None:
    failures: list[str] = []

    print("=== 1. Loading GSM8K (small sample) ===")
    train_all = load_gsm8k("train")
    test_all = load_gsm8k("test")
    train_sample = train_all[:20]
    test_sample = test_all[:20]
    print(f"  train total={len(train_all)}, test total={len(test_all)}")
    print(f"  using train_sample={len(train_sample)}, test_sample={len(test_sample)}")

    print("\n=== 2. Verifying answer extraction on known example ===")
    known = train_all[0]
    print(f"  Q: {known.question[:60]}...")
    print(f"  extracted final_answer: {known.final_answer!r}")
    if known.final_answer != "72":
        failures.append(
            f"Known-answer check failed: expected '72', got {known.final_answer!r}"
        )
    # Also spot-check the extractor directly against a synthetic string.
    synth = "Some reasoning.\n#### 1,234"
    got = extract_final_answer(synth)
    if got != "1234":
        failures.append(f"Comma-normalization failed: expected '1234', got {got!r}")

    print("\n=== 3. Rendering a few images ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered_source_texts: dict[int, str] = {}
    for ex in test_sample[:5]:
        img = render_problem_image(ex.question)
        w, h = img.size
        print(f"  idx={ex.idx} rendered {w}x{h}px")
        if w <= 0 or h <= 0 or h < 50:
            failures.append(f"idx={ex.idx} produced a degenerate image size {w}x{h}")
        img.save(OUT_DIR / f"test_{ex.idx}.png")
        rendered_source_texts[ex.idx] = ex.question
    print(f"  saved sample renders to {OUT_DIR}")

    print("\n=== 4. Leakage check (train_sample vs test_sample) ===")
    leakage = check_train_eval_leakage(train_sample, test_sample)
    print(f"  {leakage}")
    if leakage["overlap_count"] != 0:
        failures.append(f"Train/eval leakage detected: {leakage}")

    print("\n=== 5. Text/image pairing-integrity check ===")
    pairing = check_text_image_pairing(test_sample[:5], rendered_source_texts)
    print(f"  {pairing}")
    if pairing["mismatch_count"] != 0:
        failures.append(f"Text/image pairing mismatch: {pairing}")

    print("\n=== 6. Paraphrase stub: numbers must be preserved ===")
    for ex in test_sample[:5]:
        paraphrased = paraphrase_stub(ex.question, seed=0)
        orig_numbers = _all_numbers(ex.question)
        para_numbers = _all_numbers(paraphrased)
        changed = paraphrased != ex.question
        print(f"  idx={ex.idx} changed={changed} numbers_preserved={orig_numbers == para_numbers}")
        if orig_numbers != para_numbers:
            failures.append(
                f"idx={ex.idx} paraphrase changed the numbers: "
                f"{orig_numbers} -> {para_numbers}"
            )

    print("\n=== SUMMARY ===")
    if failures:
        print(f"FAIL - {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("PASS - all checks succeeded.")


if __name__ == "__main__":
    main()
