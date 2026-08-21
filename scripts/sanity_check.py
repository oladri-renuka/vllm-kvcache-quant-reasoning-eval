#!/usr/bin/env python3
"""
Manual sanity check: randomly sample and display outputs with their classifications.
For visual inspection and verification of the automated scoring.
"""

import json
import argparse
import random
from pathlib import Path


def load_results_and_scores(dtype: str, results_dir: str = "results"):
    """Load outputs and their scores for a given dtype."""
    results_file = Path(results_dir) / dtype / "outputs.jsonl"
    scores_file = Path(results_dir) / dtype / "scores.json"

    if not results_file.exists() or not scores_file.exists():
        return None, None

    # Load outputs
    outputs = {}
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                outputs[result["problem_id"]] = result

    # Load scores
    with open(scores_file, 'r') as f:
        scores = json.load(f)

    return outputs, scores


def display_sample(dtype: str, outputs: dict, scores: dict, num_samples: int = 5):
    """Display a random sample of problems with their predicted category."""

    all_scores = scores["all_scores"]
    sample_size = min(num_samples, len(all_scores))

    print("\n" + "="*80)
    print(f"SANITY CHECK: {dtype.upper()}")
    print(f"Showing {sample_size} random samples for manual review")
    print("="*80 + "\n")

    # Randomly sample
    sampled = random.sample(all_scores, sample_size)

    for i, score_entry in enumerate(sampled, 1):
        problem_id = score_entry["problem_id"]
        category = score_entry["category"]
        is_correct = score_entry["is_correct"]

        output = outputs[problem_id]
        question = output["question"]
        expected_answer = output["expected_answer"]
        generated_text = output["generated_text"]

        print(f"\n{'─'*80}")
        print(f"[Sample {i}/{sample_size}] Problem ID {problem_id}")
        print(f"Source: {output['source'].upper()}")
        print(f"Predicted Category: {category}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Correctly Answered: {'✓ YES' if is_correct else '✗ NO'}")
        print(f"{'─'*80}")

        print(f"\nQUESTION:\n{question}\n")
        print(f"GENERATED OUTPUT:\n")
        print(generated_text[:500])  # First 500 chars
        if len(generated_text) > 500:
            print(f"\n... [truncated, {len(generated_text)} total chars] ...")

        print()


def main():
    parser = argparse.ArgumentParser(description="Sanity check experiment results")
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Results directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples per dtype to review",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    dtypes = ["auto", "fp8", "int8_per_token_head"]

    print("\n" + "="*80)
    print("SANITY CHECK: Manual Review of Scored Outputs")
    print("="*80)

    for dtype in dtypes:
        print(f"\n\nLoading results for {dtype}...")
        outputs, scores = load_results_and_scores(dtype, args.results_dir)

        if outputs is None or scores is None:
            print(f"⚠ No results found for {dtype}")
            continue

        display_sample(dtype, outputs, scores, args.num_samples)

    print("\n" + "="*80)
    print("SANITY CHECK COMPLETE")
    print("Review the samples above to verify the automated classification accuracy.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
