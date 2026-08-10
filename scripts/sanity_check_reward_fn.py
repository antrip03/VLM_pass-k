import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.reward_fn import compute_score

cases = [
    ("#### format, correct", "Reasoning...\n#### 18", "18", 1.0),
    ("boxed format, correct", "Reasoning steps...\nThe answer is \\boxed{18}.", "18", 1.0),
    ("#### format, wrong answer", "Reasoning...\n#### 20", "18", 0.0),
    ("no marker at all", "Janet makes 18 dollars.", "18", 0.0),
    ("boxed format, trailing text after", "Steps...\\boxed{18} is the final answer, confirmed.", "18", 1.0),
    ("multiple #### markers, last wins", "First I said #### 5\nActually #### 18", "18", 1.0),
]

all_pass = True
for label, solution_str, ground_truth, expected in cases:
    print(f"raw solution_str repr: {solution_str!r}")
    got = compute_score("openai/gsm8k", solution_str, ground_truth, {})
    ok = got == expected
    all_pass &= ok
    print(f"{'OK  ' if ok else 'FAIL'} {label}: expected={expected}, got={got}\n")

print("ALL PASS" if all_pass else "SOME FAILED")
