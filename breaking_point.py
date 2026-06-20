"""
Breaking Point Analysis
=======================
Measures how classification accuracy degrades as the number of training
examples per class decreases. Tests n=1 up to the maximum available per class.

Usage:
    python breaking_point.py

Warning: makes 20 LLM calls per value of n tested. With n=1..4 that is 80
calls total — expect 2–3 minutes on a free Groq tier.
"""
import json
import os
from collections import defaultdict

from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import load_labeled_examples, classify_episode
from evaluate import compute_accuracy, compute_per_class_accuracy


def evaluate_with_n(n_per_class: int, labeled: list[dict], test_episodes: list[dict]) -> dict:
    by_class = defaultdict(list)
    for ex in labeled:
        by_class[ex["label"]].append(ex)

    # Take the first n examples per class (deterministic, no randomness)
    reduced = []
    for label in VALID_LABELS:
        reduced.extend(by_class[label][:n_per_class])

    predictions, ground_truth = [], []
    for episode in test_episodes:
        result = classify_episode(episode["description"], reduced)
        predictions.append(result["label"])
        ground_truth.append(episode["label"])

    return {
        "accuracy": compute_accuracy(predictions, ground_truth),
        "per_class": compute_per_class_accuracy(predictions, ground_truth),
    }


def main():
    labeled = load_labeled_examples()

    by_class = defaultdict(list)
    for ex in labeled:
        by_class[ex["label"]].append(ex)

    max_n = min(len(v) for v in by_class.values())
    counts_str = ", ".join(f"{label}={len(by_class[label])}" for label in VALID_LABELS)
    print(f"\nTraining examples available: {counts_str}")
    print(f"Testing n = 1 to {max_n} examples per class ({max_n * len(VALID_LABELS)} to {max_n * len(VALID_LABELS)} → {1 * len(VALID_LABELS)} total)\n")

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    # Header
    col = 11
    header = f"  {'n':>2}  {'overall':>8}  " + "  ".join(f"{label:>{col}}" for label in VALID_LABELS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_rows = []
    for n in range(1, max_n + 1):
        print(f"  Running n={n} ({n * len(VALID_LABELS)} examples)...", flush=True)
        r = evaluate_with_n(n, labeled, test_episodes)
        per = r["per_class"]
        row_data = {"n": n, "overall": r["accuracy"], "per_class": per}
        all_rows.append(row_data)
        class_cols = "  ".join(f"{per[label]['accuracy']:>{col}.0%}" for label in VALID_LABELS)
        print(f"  {n:>2}  {r['accuracy']:>8.0%}  {class_cols}")

    # Analysis
    print("\n--- Observations ---")
    baseline_acc = all_rows[-1]["overall"]
    for row in all_rows:
        drop = baseline_acc - row["overall"]
        if drop > 0:
            worst_label = min(VALID_LABELS, key=lambda l: row["per_class"][l]["accuracy"])
            worst_acc = row["per_class"][worst_label]["accuracy"]
            print(f"  n={row['n']}: overall={row['overall']:.0%}  ({drop:.0%} below full)  "
                  f"weakest class: {worst_label} at {worst_acc:.0%}")
        else:
            print(f"  n={row['n']}: overall={row['overall']:.0%}  (same as full training set)")


if __name__ == "__main__":
    main()
