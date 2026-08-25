"""
Truncation control for the §5.4 modality-dependent compliance claim
(mentor review, 2026-08-25, item 1 - "the one issue that could kill the
contribution").

THE THREAT
----------
"format_compliance_rate" as computed everywhere in this project is a
PARSE rate: 1 - unparsed_rate. The mentor's objection: an unparsed
completion has at least three distinct causes, and only one of them is
instruction-following:

  1. The model ignored the format instruction (what §5.4 claims).
  2. Generation hit max_new_tokens (1200) before reaching the answer.
  3. The model did something else entirely - e.g. described the image
     instead of solving it, plausible specifically in condition E.

Cause 2 is the dangerous one. If RL-tuned outputs are simply SHORTER
(a well-documented RLVR effect independent of this project - Dr. GRPO's
own length-bias fix targets exactly this) and base-model completions run
longer especially when reading from an image, then the "compliance gap"
in E could be a LENGTH/TRUNCATION effect, not an instruction-following
effect. That would swap the paper's actual contribution (modality
degrades instruction-following) for an already-known one (RL shortens
outputs), and specifically undermines the §5.4 table this project has
built its novelty claim on.

WHAT THIS SCRIPT DOES
----------------------
Free - CPU-only, no GPU, uses only saved completion text.

1. Retokenizes every unparsed completion with the real Qwen2.5-VL
   tokenizer (chat template excluded - just the completion text) and
   flags TRUNCATED = token count >= cap - margin. The cap is 1200
   (PLAN.md Section 3 Phase 1 item 7); a small margin accounts for the
   tokenizer not being byte-identical to what vLLM/veRL counted at
   generation time.
2. Reports truncation rate by model x condition - the direct test of
   cause 2.
3. Recomputes "format compliance" EXCLUDING truncated generations. If
   the E > T compliance gap survives on EOS-terminated generations only,
   the mechanism is instruction-following, as claimed. If it collapses,
   the claim needs to change to a length/truncation effect.
4. Runs a cheap keyword heuristic over base-model condition-E unparsed
   completions for cause 3 (described the image instead of solving),
   reported as a fraction with the actual matched snippets so it can be
   spot-checked rather than trusted blindly.

USAGE
    python3 scripts/run_truncation_control.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
GENERATION_CAP = 1200
# Real generations are decoded, then re-encoded here with a DIFFERENT
# tokenizer call than the one that produced them (no chat-template
# context, no special tokens). This can differ from the true generation
# token count by a handful of tokens. A margin of 15 keeps the flag
# conservative - it is far better to under-flag borderline cases as
# "not truncated" than to inflate the truncation rate and manufacture a
# finding that is not there.
TRUNCATION_MARGIN = 15

# Cheap, deliberately narrow heuristic for "described the image instead
# of solving it" - phrases a model uses to describe rather than compute.
# Matched snippets are printed so this can be spot-checked, not trusted
# as a ground-truth classifier.
_DESCRIBE_RE = re.compile(
    r"\b(the image (shows|depicts|contains)|this (image|picture) (shows|depicts)|"
    r"i (can\s+)?see (a|an|the)|the (picture|photo) (shows|depicts))\b",
    re.I,
)


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def token_count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    print("Loading tokenizer (CPU only, no GPU)...", flush=True)
    tok = load_tokenizer()

    report = {"generation_cap": GENERATION_CAP, "truncation_margin": TRUNCATION_MARGIN}

    print("\n=== 1. Truncation rate by model x condition (unparsed generations only) ===")
    print(f"{'model':<6}{'cond':<5}{'unparsed':>10}{'truncated':>11}{'trunc%':>9}{'compliance (raw)':>19}"
          f"{'compliance (EOS-only)':>24}")
    print("-" * 84)

    dfs = {kind: pd.read_parquet(f"eval_results/records_{kind}.fallback.parquet") for kind in ("base", "rl")}
    report["by_condition"] = {}

    for kind in ("base", "rl"):
        df = dfs[kind]
        for cond in ("T", "D", "E"):
            sub = df[df.condition == cond]
            total = len(sub)
            unparsed = sub[sub.extracted_answer.isna()]
            n_unparsed = len(unparsed)

            if n_unparsed == 0:
                trunc_n, trunc_pct = 0, 0.0
            else:
                lengths = [token_count(tok, c) for c in unparsed["completion"].astype(str)]
                truncated_flags = [l >= GENERATION_CAP - TRUNCATION_MARGIN for l in lengths]
                trunc_n = sum(truncated_flags)
                trunc_pct = 100 * trunc_n / n_unparsed
                unparsed = unparsed.assign(_truncated=truncated_flags)

            raw_compliance = 1 - n_unparsed / total
            # EOS-only compliance: treat a truncated generation as
            # "we don't know" rather than "non-compliant" - exclude it
            # from both numerator and denominator.
            n_truncated = trunc_n if n_unparsed else 0
            eos_denom = total - n_truncated
            eos_numer = total - n_unparsed  # parsed count is unaffected
            eos_compliance = eos_numer / eos_denom if eos_denom else float("nan")

            print(f"{kind:<6}{cond:<5}{n_unparsed:>10}{trunc_n:>11}{trunc_pct:>8.1f}%"
                  f"{raw_compliance:>19.4f}{eos_compliance:>24.4f}")

            report["by_condition"][f"{kind}/{cond}"] = {
                "total": total,
                "unparsed": n_unparsed,
                "truncated_among_unparsed": trunc_n,
                "truncated_pct_of_unparsed": round(trunc_pct, 2),
                "compliance_raw": round(raw_compliance, 4),
                "compliance_eos_only": round(eos_compliance, 4) if eos_denom else None,
            }

    print("\n=== 2. Does the E > T compliance GAP survive on EOS-terminated generations only? ===")
    gap_raw = {}
    gap_eos = {}
    for cond in ("T", "D", "E"):
        b = report["by_condition"][f"base/{cond}"]
        r = report["by_condition"][f"rl/{cond}"]
        gap_raw[cond] = r["compliance_raw"] - b["compliance_raw"]
        if b["compliance_eos_only"] is not None and r["compliance_eos_only"] is not None:
            gap_eos[cond] = r["compliance_eos_only"] - b["compliance_eos_only"]
    print(f"{'cond':<6}{'gap (raw)':>12}{'gap (EOS-only)':>16}")
    for cond in ("T", "D", "E"):
        eos_val = gap_eos.get(cond)
        print(f"{cond:<6}{gap_raw[cond]:>+12.4f}{(f'{eos_val:+.4f}' if eos_val is not None else 'n/a'):>16}")

    if "E" in gap_eos and "T" in gap_eos:
        e_survives = gap_eos["E"] > gap_eos["T"]
        verdict = (
            f"E-vs-T compliance gap {'SURVIVES' if e_survives else 'DOES NOT SURVIVE'} on "
            f"EOS-only generations (E={gap_eos['E']:+.4f} vs T={gap_eos['T']:+.4f} raw; "
            f"E={gap_eos.get('E'):+.4f} vs T={gap_eos.get('T'):+.4f} EOS-only)."
        )
    else:
        verdict = "Insufficient EOS-only data to compare."
    print(f"\nVERDICT: {verdict}")
    report["verdict"] = verdict

    print("\n=== 3. 'Described the image instead of solving it' - base model, condition E only ===")
    base_e_unparsed = dfs["base"][(dfs["base"].condition == "E") & (dfs["base"].extracted_answer.isna())]
    matches = []
    for _, row in base_e_unparsed.iterrows():
        m = _DESCRIBE_RE.search(str(row["completion"]))
        if m:
            matches.append((int(row["problem_idx"]), m.group(0), str(row["completion"])[:160]))
    frac = len(matches) / max(len(base_e_unparsed), 1)
    print(f"  {len(matches)}/{len(base_e_unparsed)} unparsed base/E completions match a "
          f"'described rather than solved' heuristic ({frac:.1%})")
    print("  NOTE: this is a narrow keyword heuristic for triage, not a validated classifier.")
    for pidx, hit, snippet in matches[:5]:
        print(f"    problem {pidx}: matched {hit!r} in: {snippet!r}")
    report["describes_instead_of_solves"] = {
        "matched": len(matches),
        "total_unparsed_base_e": len(base_e_unparsed),
        "fraction": round(frac, 4),
        "sample_matches": [{"problem_idx": p, "match": h, "snippet": s} for p, h, s in matches[:10]],
    }

    Path("eval_results/truncation_control.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nWrote eval_results/truncation_control.json")


if __name__ == "__main__":
    main()
