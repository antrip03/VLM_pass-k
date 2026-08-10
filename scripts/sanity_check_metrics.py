"""
Local (no GPU/Modal needed) sanity check for src/metrics/pass_at_k.py and
src/metrics/answer_extraction.py - pure logic, verified against
hand-computed reference values before anything depends on it.

Usage: python scripts/sanity_check_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controls.headroom_check import check_headroom
from src.metrics.answer_extraction import extract_model_answer, is_correct
from src.metrics.bootstrap_ci import bootstrap_delta_ci, bootstrap_pass_at_k_ci
from src.metrics.pass_at_k import mean_pass_at_k, pass_at_k
from src.metrics.transcription_fidelity import (
    character_level_similarity,
    is_exact_transcription_match,
    mean_transcription_fidelity,
)


def main() -> None:
    failures: list[str] = []

    print("=== 1. pass_at_k against hand-computed reference values ===")

    # pass@1 must always equal c/n exactly (a known identity of the
    # unbiased estimator).
    r = pass_at_k(n=10, c=5, k=1)
    print(f"  n=10, c=5, k=1 -> {r} (expect 0.5)")
    if abs(r - 0.5) > 1e-9:
        failures.append(f"pass@1 identity failed: got {r}")

    # All correct -> pass@k = 1.0 for any k.
    r = pass_at_k(n=10, c=10, k=7)
    print(f"  n=10, c=10, k=7 -> {r} (expect 1.0)")
    if abs(r - 1.0) > 1e-9:
        failures.append(f"all-correct case failed: got {r}")

    # All incorrect -> pass@k = 0.0 for any k <= n.
    r = pass_at_k(n=10, c=0, k=5)
    print(f"  n=10, c=0, k=5 -> {r} (expect 0.0)")
    if abs(r - 0.0) > 1e-9:
        failures.append(f"all-incorrect case failed: got {r}")

    # Hand-computed via combinatorics: n=5, c=2, k=3 ->
    # 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9
    r = pass_at_k(n=5, c=2, k=3)
    print(f"  n=5, c=2, k=3 -> {r} (expect 0.9)")
    if abs(r - 0.9) > 1e-9:
        failures.append(f"hand-computed C(3,3)/C(5,3) case failed: got {r}")

    # n-c < k boundary: n=5, c=2 (n-c=3), k=5 -> guaranteed 1.0
    r = pass_at_k(n=5, c=2, k=5)
    print(f"  n=5, c=2, k=5 (n-c < k) -> {r} (expect 1.0)")
    if abs(r - 1.0) > 1e-9:
        failures.append(f"n-c<k boundary case failed: got {r}")

    print("\n=== 2. pass_at_k input validation ===")
    try:
        pass_at_k(n=5, c=2, k=10)  # k > n
        failures.append("Expected ValueError when k > n")
    except ValueError:
        print("  k > n correctly raises ValueError")

    try:
        pass_at_k(n=5, c=8, k=1)  # c > n
        failures.append("Expected ValueError when c > n")
    except ValueError:
        print("  c > n correctly raises ValueError")

    print("\n=== 3. mean_pass_at_k ===")
    # Two problems: one always correct (pass@1=1.0), one always wrong (pass@1=0.0)
    results = [(10, 10), (10, 0)]
    m = mean_pass_at_k(results, k=1)
    print(f"  [(10,10), (10,0)] at k=1 -> {m} (expect 0.5)")
    if abs(m - 0.5) > 1e-9:
        failures.append(f"mean_pass_at_k failed: got {m}")

    print("\n=== 4. answer_extraction ===")
    cases = [
        ("The answer is #### 72", "72"),
        ("Reasoning...\n#### 1,234", "1234"),
        ("#### -5", "-5"),
        ("#### 3.5", "3.5"),
        ("No marker here at all", None),
        # \boxed{} fallback - the exact failure pattern found live via
        # scripts/inspect_raw_generations.py: the model reasoned correctly
        # to 18 and wrote \boxed{18}, which the original "#### only"
        # extractor completely missed, marking a correct completion wrong.
        ("So the answer is:\n\\[\n\\boxed{18}\n\\]", "18"),
        ("\\boxed{1,234}", "1234"),
        # #### must still win when both are present (stated priority order).
        ("\\boxed{5}\n#### 10", "10"),
        # Known, documented residual gap: plain prose with neither marker.
        ("Therefore, Janet makes $18 every day at the market.", None),
    ]
    for text, expected in cases:
        got = extract_model_answer(text)
        print(f"  {text!r} -> {got!r} (expect {expected!r})")
        if got != expected:
            failures.append(f"extract_model_answer({text!r}) gave {got!r}, expected {expected!r}")

    print("\n=== 5. is_correct ===")
    if not is_correct("blah blah #### 72", "72"):
        failures.append("is_correct should be True for matching answer")
    if is_correct("blah blah #### 71", "72"):
        failures.append("is_correct should be False for mismatched answer")
    if is_correct("no marker", "72"):
        failures.append("is_correct should be False when no answer extracted")
    print("  matching/mismatched/missing cases all behave correctly")

    print("\n=== 6. check_headroom ===")
    # IMPORTANT, and this is exactly why this test exists: n must be >>
    # k_high, not just >= it. A first version of this test used n=20 with
    # k_high=16 (k/n = 0.8) and found that even a genuinely weak model
    # (1 correct out of 20) showed an inflated pass@16 - drawing 80% of
    # your samples will almost always catch a single rare correct answer,
    # regardless of whether the model has real "headroom." That's a
    # property of the combinatorics at that n/k ratio, not a bug in
    # pass_at_k (already validated separately against hand-computed
    # values above) - it's why PLAN.md requires n >> k, not just n >= k.
    # Using n=64 here (k/n = 0.25) for a realistic test.

    # Real headroom: rarely right first try, usually right within 16 tries.
    headroom_results = [(64, 2)] * 10 + [(64, 45)] * 10
    report = check_headroom(headroom_results, k_high=16)
    print(f"  clear-headroom case (n=64) -> {report}")
    if not report["has_headroom"]:
        failures.append(f"Expected has_headroom=True for a clear gap: {report}")

    # No real headroom: unambiguously zero correct, ever, on any problem -
    # a clean test of this pathway rather than a mix that happens to
    # straddle the default threshold (a (64,1) minority would put the
    # gap right at the boundary, which tests the threshold constant more
    # than the underlying logic).
    no_headroom_results = [(64, 0)] * 20
    report2 = check_headroom(no_headroom_results, k_high=16)
    print(f"  no-headroom case (n=64, all zero) -> {report2}")
    if report2["has_headroom"]:
        failures.append(f"Expected has_headroom=False for a zero gap: {report2}")
    if report2["gap"] != 0.0:
        failures.append(f"Expected exactly zero gap when nothing is ever correct: {report2}")

    try:
        check_headroom([], k_high=16)
        failures.append("Expected ValueError for empty results_per_problem")
    except ValueError:
        print("  empty input correctly raises ValueError")

    print("\n=== 7. bootstrap_pass_at_k_ci ===")
    # Zero-variance case: every problem identical (10/10 correct) -> every
    # bootstrap resample is also 10/10 correct, so the CI must collapse
    # to exactly the point estimate (no spread possible).
    zero_var_results = [(10, 10)] * 20
    ci = bootstrap_pass_at_k_ci(zero_var_results, k=1, n_bootstrap=500, seed=0)
    print(f"  zero-variance case -> {ci} (expect point/lower/upper all 1.0)")
    if not (abs(ci["point_estimate"] - 1.0) < 1e-9 and abs(ci["ci_lower"] - 1.0) < 1e-9 and abs(ci["ci_upper"] - 1.0) < 1e-9):
        failures.append(f"bootstrap_pass_at_k_ci zero-variance case failed: {ci}")

    # Mixed case: half the problems always correct, half always wrong.
    # Point estimate must be exactly 0.5 (deterministic mean); the CI
    # must have real width (not collapse to a point) and must bracket 0.5.
    mixed_results = [(10, 10)] * 10 + [(10, 0)] * 10
    ci2 = bootstrap_pass_at_k_ci(mixed_results, k=1, n_bootstrap=2000, seed=0)
    print(f"  mixed case -> {ci2} (expect point=0.5, lower<0.5<upper)")
    if abs(ci2["point_estimate"] - 0.5) > 1e-9:
        failures.append(f"bootstrap_pass_at_k_ci mixed-case point estimate wrong: {ci2}")
    if not (ci2["ci_lower"] < 0.5 < ci2["ci_upper"]):
        failures.append(f"bootstrap_pass_at_k_ci mixed-case CI should bracket 0.5: {ci2}")

    print("\n=== 8. bootstrap_delta_ci ===")
    # Identical A and B (same list object even) -> every paired resample
    # gives delta=0 exactly, so the CI must collapse to [0, 0] and NOT be
    # flagged significant (there is no real difference to detect).
    identical = [(10, 5)] * 15
    delta_ci_same = bootstrap_delta_ci(identical, identical, k=1, n_bootstrap=500, seed=0)
    print(f"  identical A/B -> {delta_ci_same} (expect point/lower/upper all 0.0, significant=False)")
    if not (
        abs(delta_ci_same["point_estimate"]) < 1e-9
        and abs(delta_ci_same["ci_lower"]) < 1e-9
        and abs(delta_ci_same["ci_upper"]) < 1e-9
    ):
        failures.append(f"bootstrap_delta_ci identical-inputs case failed: {delta_ci_same}")
    if delta_ci_same["significant"]:
        failures.append(f"bootstrap_delta_ci identical inputs should not be significant: {delta_ci_same}")

    # Maximally, uniformly different A (always correct) vs B (always
    # wrong) -> zero variance again, delta collapses to exactly 1.0, and
    # this real difference MUST be flagged significant.
    all_correct = [(10, 10)] * 15
    all_wrong = [(10, 0)] * 15
    delta_ci_diff = bootstrap_delta_ci(all_correct, all_wrong, k=1, n_bootstrap=500, seed=0)
    print(f"  maximally different A/B -> {delta_ci_diff} (expect point/lower/upper all 1.0, significant=True)")
    if abs(delta_ci_diff["point_estimate"] - 1.0) > 1e-9:
        failures.append(f"bootstrap_delta_ci max-difference point estimate wrong: {delta_ci_diff}")
    if not delta_ci_diff["significant"]:
        failures.append(f"bootstrap_delta_ci max-difference case should be significant: {delta_ci_diff}")

    try:
        bootstrap_delta_ci([(10, 5)] * 5, [(10, 5)] * 3, k=1)  # mismatched lengths
        failures.append("Expected ValueError for mismatched results_a/results_b lengths")
    except ValueError:
        print("  mismatched-length inputs correctly raise ValueError")

    print("\n=== 9. transcription_fidelity ===")
    # Exact match after normalization - including the real curly-quote
    # character GSM8K questions actually contain ("Janet's ducks..."),
    # not a synthetic example.
    if not is_exact_transcription_match("Janet’s ducks lay 16 eggs", "janet's ducks lay 16 eggs"):
        failures.append("is_exact_transcription_match should normalize curly quotes + case")
    if not is_exact_transcription_match("  Extra   whitespace   here  ", "Extra whitespace here"):
        failures.append("is_exact_transcription_match should normalize whitespace")
    if is_exact_transcription_match("The robe needs 2 bolts", "The robe needs 3 bolts"):
        failures.append("is_exact_transcription_match should NOT match genuinely different text")
    print("  exact-match normalization (curly quotes, whitespace, real mismatch) all correct")

    sim_identical = character_level_similarity("same text here", "same text here")
    if abs(sim_identical - 1.0) > 1e-9:
        failures.append(f"character_level_similarity of identical strings should be 1.0, got {sim_identical}")
    sim_one_char_off = character_level_similarity("the answer is 18", "the answer is 19")
    print(f"  one-digit-off similarity -> {sim_one_char_off} (expect high but < 1.0)")
    if not (0.9 < sim_one_char_off < 1.0):
        failures.append(f"character_level_similarity for a 1-char difference looked wrong: {sim_one_char_off}")

    fidelity_pairs = [
        ("Janet's ducks lay 16 eggs", "Janet's ducks lay 16 eggs"),  # exact
        ("A robe needs 2 bolts of blue fiber", "A robe needs 2 bolts of red fiber"),  # close but wrong
        ("completely unrelated text", "Josh buys a house for $80,000"),  # unrelated
    ]
    fidelity = mean_transcription_fidelity(fidelity_pairs)
    print(f"  mean_transcription_fidelity over 3 pairs (1 exact, 2 not) -> {fidelity}")
    if abs(fidelity["exact_match_rate"] - (1 / 3)) > 1e-9:
        failures.append(f"mean_transcription_fidelity exact_match_rate wrong: {fidelity}")
    if fidelity["n_samples"] != 3:
        failures.append(f"mean_transcription_fidelity n_samples wrong: {fidelity}")

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
