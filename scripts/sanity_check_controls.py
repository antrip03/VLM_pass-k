"""
Local (no GPU/Modal needed) sanity check for src/controls/truncation_check.py
- pure logic, testable with synthetic data before Step 6's real sampling
harness exists to produce real generations.

The manipulation check (src/controls/manipulation_check.py) needs a real
model + GPU and is validated separately by
scripts/validate_manipulation_check_on_modal.py.

Usage: python scripts/sanity_check_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.truncation_check import (
    GenerationRecord,
    check_truncation_symmetry,
    compute_truncation_rate,
    hit_max_tokens,
)


def main() -> None:
    failures: list[str] = []

    print("=== 1. hit_max_tokens ===")
    # Stopped naturally, well before the cap.
    r1 = hit_max_tokens([1, 2, 3, 50257], max_new_tokens=500, eos_token_id=50257)
    print(f"  short output ending in EOS -> hit_max_tokens={r1} (expect False)")
    if r1 is not False:
        failures.append("Expected False for a short, EOS-terminated output")

    # Hit the cap, last token is NOT eos -> truncated.
    truncated_output = list(range(500))  # 500 tokens, none of them EOS (50257)
    r2 = hit_max_tokens(truncated_output, max_new_tokens=500, eos_token_id=50257)
    print(f"  500 tokens, no EOS -> hit_max_tokens={r2} (expect True)")
    if r2 is not True:
        failures.append("Expected True for a full-length, non-EOS-terminated output")

    # Hit the cap, but the *last* token happens to be EOS - generous case.
    edge_output = list(range(499)) + [50257]
    r3 = hit_max_tokens(edge_output, max_new_tokens=500, eos_token_id=50257)
    print(f"  500 tokens, last is EOS -> hit_max_tokens={r3} (expect False)")
    if r3 is not False:
        failures.append("Expected False when the final token at the cap is EOS")

    # Multiple valid eos ids (list form).
    r4 = hit_max_tokens(list(range(500)), max_new_tokens=500, eos_token_id=[50257, 50258])
    print(f"  list-form eos_token_id, no match -> hit_max_tokens={r4} (expect True)")
    if r4 is not True:
        failures.append("Expected True with list-form eos_token_id and no match")

    print("\n=== 2. compute_truncation_rate ===")
    records = [
        GenerationRecord(hit_cap=False, answer_extracted=True),
        GenerationRecord(hit_cap=True, answer_extracted=True),  # truncated but still got an answer
        GenerationRecord(hit_cap=True, answer_extracted=False),  # the real problem case
        GenerationRecord(hit_cap=False, answer_extracted=False),  # not truncated, just wrong
    ]
    report = compute_truncation_rate(records)
    print(f"  {report}")
    if report["truncated_no_answer"] != 1 or report["n"] != 4:
        failures.append(f"compute_truncation_rate gave unexpected counts: {report}")
    if abs(report["truncation_rate"] - 0.25) > 1e-9:
        failures.append(f"compute_truncation_rate gave unexpected rate: {report}")

    empty_report = compute_truncation_rate([])
    print(f"  empty input -> {empty_report}")
    if empty_report["n"] != 0:
        failures.append("compute_truncation_rate should handle an empty list cleanly")

    print("\n=== 3. check_truncation_symmetry ===")
    clean_rates = {"base_T": 0.01, "rl_T": 0.02, "base_E": 0.03, "rl_E": 0.04}
    clean_report = check_truncation_symmetry(clean_rates)
    print(f"  low, symmetric rates -> clean={clean_report['clean']} (expect True)")
    if not clean_report["clean"]:
        failures.append(f"Expected clean=True for low symmetric rates: {clean_report}")

    asymmetric_rates = {"base_T": 0.01, "rl_T": 0.02, "base_E": 0.02, "rl_E": 0.30}
    asym_report = check_truncation_symmetry(asymmetric_rates)
    print(f"  asymmetric rl_E rate -> clean={asym_report['clean']} (expect False)")
    print(f"    high_truncation_cells={asym_report['high_truncation_cells']}")
    print(f"    base_vs_rl_asymmetries={asym_report['base_vs_rl_asymmetries']}")
    if asym_report["clean"]:
        failures.append("Expected clean=False when rl_E truncation is asymmetrically high")
    if "E" not in asym_report["base_vs_rl_asymmetries"]:
        failures.append("Expected the E condition to be flagged as asymmetric")

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
