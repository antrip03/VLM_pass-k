"""
Transcription fidelity (PLAN.md Section 4 item 4, Condition D only):
compares the model's own self-transcription of the problem image against
the real ground-truth question text - isolates pure perception/reading
error from downstream reasoning error, and gives a directly reportable
"transcription accuracy" metric independent of whether the model then
solved the problem correctly.
"""

from __future__ import annotations

import re


def normalize_for_comparison(text: str) -> str:
    """
    Normalizes whitespace and common unicode punctuation variants before
    comparison - a transcription that differs only in curly vs. straight
    quotes or extra whitespace should not count as a transcription
    failure (that's an encoding artifact, not a perception error).
    """
    text = text.strip().lower()
    text = text.replace("’", "'").replace("‘", "'")  # curly single quotes
    text = text.replace("“", '"').replace("”", '"')  # curly double quotes
    text = re.sub(r"\s+", " ", text)  # collapse all whitespace runs to one space
    return text


def is_exact_transcription_match(transcription: str, ground_truth_question: str) -> bool:
    """Exact match after normalization - the strict fidelity check."""
    return normalize_for_comparison(transcription) == normalize_for_comparison(ground_truth_question)


def _levenshtein_distance(a: str, b: str) -> int:
    """Standard O(len(a)*len(b)) DP edit distance - no external dependency."""
    if len(a) < len(b):
        a, b = b, a
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row[j] = min(insertions, deletions, substitutions)
        previous_row = current_row
    return previous_row[-1]


def character_level_similarity(transcription: str, ground_truth_question: str) -> float:
    """
    A softer, secondary fidelity measure: normalized edit-distance
    similarity in [0, 1] (1.0 = identical, 0.0 = maximally different).
    Useful because a transcription that's correct except for one
    misread digit is a meaningfully different failure mode than a
    transcription unrelated to the image - exact match alone collapses
    that distinction into the same "wrong" bucket.
    """
    a = normalize_for_comparison(transcription)
    b = normalize_for_comparison(ground_truth_question)
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(a, b)
    return 1.0 - (distance / max_len)


def mean_transcription_fidelity(transcriptions_and_ground_truths: list[tuple[str, str]]) -> dict:
    """
    Aggregate transcription fidelity across many (transcription,
    ground_truth_question) pairs - e.g. every Condition D sample for one
    model/checkpoint.
    """
    if not transcriptions_and_ground_truths:
        raise ValueError("transcriptions_and_ground_truths must not be empty")
    exact_matches = [is_exact_transcription_match(t, g) for t, g in transcriptions_and_ground_truths]
    similarities = [character_level_similarity(t, g) for t, g in transcriptions_and_ground_truths]
    return {
        "exact_match_rate": sum(exact_matches) / len(exact_matches),
        "mean_char_similarity": sum(similarities) / len(similarities),
        "n_samples": len(transcriptions_and_ground_truths),
    }
