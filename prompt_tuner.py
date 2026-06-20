"""
Prompt Tuning Experiment
========================
Tests three different strategies for presenting training examples and measures
the effect on per-class accuracy. Useful for understanding what the LLM is
sensitive to in the few-shot prompt.

The three variants:
  A (baseline)  – current implementation; examples in their natural order
  B (recency)   – target class examples placed last (LLMs weight recent context)
  C (upsampled) – target class examples repeated twice (doubled representation)

Usage:
    python prompt_tuner.py                   # targets the first class in VALID_LABELS
    python prompt_tuner.py --class panel     # targets "panel"
    python prompt_tuner.py --class narrative # targets "narrative"

Warning: runs 3 full evaluations (3 × 20 = 60 LLM calls). ~2 minutes on Groq.
"""
import json
import os
import re
import sys
from collections import defaultdict

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import load_labeled_examples, build_few_shot_prompt
from evaluate import compute_accuracy, compute_per_class_accuracy


_client = Groq(api_key=GROQ_API_KEY)


def _call_llm(prompt: str) -> dict:
    """Send prompt to LLM, parse JSON response, return {label, confidence}."""
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(text)
            label = parsed.get("label", "").strip().lower()
            confidence = parsed.get("confidence")
        except json.JSONDecodeError:
            m = re.search(r'"label"\s*:\s*"(\w+)"', text)
            label = m.group(1).lower() if m else ""
            confidence = None
        if label not in VALID_LABELS:
            label = "unknown"
        return {"label": label, "confidence": confidence}
    except Exception:
        return {"label": "unknown", "confidence": None}


def _run_variant(labeled: list[dict], test_episodes: list[dict], label: str = "variant") -> dict:
    """Evaluate all test episodes using the given labeled examples as few-shot context."""
    predictions, ground_truth = [], []
    for episode in test_episodes:
        prompt = build_few_shot_prompt(labeled, episode["description"])
        result = _call_llm(prompt)
        predictions.append(result["label"])
        ground_truth.append(episode["label"])
    return {
        "accuracy": compute_accuracy(predictions, ground_truth),
        "per_class": compute_per_class_accuracy(predictions, ground_truth),
    }


def variant_a(labeled: list[dict]) -> list[dict]:
    """Baseline: natural order."""
    return labeled


def variant_b(labeled: list[dict], target_class: str) -> list[dict]:
    """Recency-biased: target class examples appear last."""
    non_target = [ex for ex in labeled if ex["label"] != target_class]
    target = [ex for ex in labeled if ex["label"] == target_class]
    return non_target + target


def variant_c(labeled: list[dict], target_class: str) -> list[dict]:
    """Upsampled: target class examples repeated twice."""
    target = [ex for ex in labeled if ex["label"] == target_class]
    return labeled + target


def main():
    target_class = VALID_LABELS[0]
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--class" and i + 1 < len(sys.argv[1:]):
            target_class = sys.argv[i + 2]
        elif arg.startswith("--class="):
            target_class = arg.split("=", 1)[1]
        elif arg in VALID_LABELS:
            target_class = arg

    if target_class not in VALID_LABELS:
        print(f"Unknown class '{target_class}'. Valid labels: {VALID_LABELS}")
        sys.exit(1)

    labeled = load_labeled_examples()
    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    variants = [
        ("A (baseline)",  variant_a(labeled)),
        ("B (recency)",   variant_b(labeled, target_class)),
        ("C (upsampled)", variant_c(labeled, target_class)),
    ]

    print(f"\nPrompt Tuning — target class: {target_class}")
    print(f"{'='*60}\n")

    all_results = []
    for name, examples in variants:
        n_target = sum(1 for ex in examples if ex["label"] == target_class)
        print(f"Running variant {name} ({len(examples)} examples, {n_target} × {target_class})...", flush=True)
        r = _run_variant(examples, test_episodes)
        all_results.append((name, r))

    # Results table
    col = 12
    print(f"\n{'Variant':<20}  {'overall':>8}  " + "  ".join(f"{l:>{col}}" for l in VALID_LABELS))
    print("-" * (20 + 2 + 8 + 2 + col * len(VALID_LABELS) + 2 * (len(VALID_LABELS) - 1)))
    for name, r in all_results:
        per = r["per_class"]
        class_cols = "  ".join(f"{per[l]['accuracy']:>{col}.0%}" for l in VALID_LABELS)
        print(f"{name:<20}  {r['accuracy']:>8.0%}  {class_cols}")

    # Delta vs baseline
    baseline_per = all_results[0][1]["per_class"]
    print(f"\nDelta vs baseline for '{target_class}':")
    for name, r in all_results[1:]:
        delta = r["per_class"][target_class]["accuracy"] - baseline_per[target_class]["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"  {name}: {sign}{delta:.0%}")


if __name__ == "__main__":
    main()
