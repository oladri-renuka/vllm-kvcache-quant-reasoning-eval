#!/usr/bin/env python3
"""
Score outputs using the 6-category taxonomy for reasoning failures.

Categories:
1. No Failure: Correct answer + coherent reasoning
2. Shortcut Collapse: Correct answer but reasoning is incomplete/skipped/vague
3. Premise Hijacking: Accepts false assumption, reasons correctly from it
4. Confidence Snowballing: Single early error propagates through solution
5. Overcounting: Correct intermediate result, then continues unnecessarily
6. Incoherent/Garbled: Output is unreadable or indicates crash
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def normalize_answer(answer_str: str) -> str:
    """Normalize answer for comparison."""
    # Remove extra whitespace
    answer = answer_str.strip().lower()
    # Try to extract numbers/fractions
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def extract_final_number(text: str) -> List[str]:
    """Extract candidate final answers from text."""
    # Look for patterns like "answer is X", "final answer: X", "=", etc.
    candidates = []

    # Pattern: "answer is ... "
    matches = re.findall(r'answer[:\s]+([0-9./\-\w]+)', text, re.IGNORECASE)
    candidates.extend(matches)

    # Pattern: "final answer"
    matches = re.findall(r'final[:\s]+([0-9./\-\w]+)', text, re.IGNORECASE)
    candidates.extend(matches)

    # Pattern: "= X" (end of line)
    matches = re.findall(r'=\s*([0-9./\-\w]+)(?:\n|$)', text, re.IGNORECASE)
    candidates.extend(matches)

    # Pattern: numbers followed by period at end
    matches = re.findall(r'([0-9./\-\w]+)\s*\.?\s*$', text.strip())
    candidates.extend(matches)

    return list(set(candidates))


def check_answer_correctness(generated_text: str, expected_answer: str) -> bool:
    """
    Check if the generated output contains the expected answer.
    This is a heuristic check; implement more sophisticated matching as needed.
    """
    # Normalize both
    expected_norm = normalize_answer(expected_answer)
    generated_norm = normalize_answer(generated_text)

    # Direct substring match
    if expected_norm in generated_norm:
        return True

    # Extract numbers and compare
    expected_nums = re.findall(r'[\d.]+', expected_norm)
    generated_nums = re.findall(r'[\d.]+', generated_norm)

    if expected_nums and generated_nums:
        # Check if any match
        for exp in expected_nums:
            if exp in generated_nums:
                return True

    return False


def detect_shortcut_collapse(text: str) -> bool:
    """
    Detect Shortcut Collapse: correct answer but reasoning is vague/incomplete.
    Heuristic: Very short output or missing step-by-step explanation.
    """
    # If text is very short, likely a shortcut
    if len(text) < 100:
        return True

    # Check for absence of reasoning markers
    reasoning_indicators = [
        r'let\s+(?:me\s+)?(?:think|work|solve)',
        r'step\s+\d+',
        r'therefore',
        r'so\s+(?:we\s+)?(?:have|get)',
        r'this\s+(?:means|gives|implies)',
        r'notice\s+that',
        r'(?:by|using)\s+(?:the|a)',
    ]

    found_reasoning = any(re.search(pattern, text, re.IGNORECASE) for pattern in reasoning_indicators)

    # If no reasoning found, likely shortcut collapse
    return not found_reasoning


def detect_premise_hijacking(text: str, question: str) -> bool:
    """
    Detect Premise Hijacking: model accepts a false assumption and reasons correctly from it.
    Heuristic: Look for contradictions with the problem statement.
    """
    # This is hard to detect without semantic understanding.
    # For now, check for explicit contradictions or assumptions not in the question.
    hijacking_patterns = [
        r'(?:assume|let|suppose)\s+(?:the\s+)?(?:answer|result)\s+is',  # Circular reasoning
        r'assume\s+(?:the|a)\s+(?:different|wrong|false)',
    ]

    found = any(re.search(pattern, text, re.IGNORECASE) for pattern in hijacking_patterns)
    return found


def detect_confidence_snowballing(text: str) -> bool:
    """
    Detect Confidence Snowballing: single early error propagates.
    Heuristic: Look for calculations that seem to follow logically but start from wrong premise.
    """
    # This is very hard to detect without actually executing the math.
    # Heuristic: if many calculations but final answer seems wrong relative to problem scale
    calc_count = len(re.findall(r'[\+\-\*/]', text))
    line_count = len(text.split('\n'))

    # Many calculations spread over few lines might indicate cascading errors
    if calc_count > 10 and line_count < 5:
        return True

    return False


def detect_overcounting(text: str, question: str) -> bool:
    """
    Detect Overcounting: correct intermediate result, then continues unnecessarily.
    Heuristic: Look for multiple answers or "but then" continuations.
    """
    # Check for patterns like "the answer is X, but" or multiple final answers
    multiple_answers = len(re.findall(r'(?:the\s+)?answer\s+is', text, re.IGNORECASE)) > 1
    unnecessary_continuation = bool(re.search(r'(?:but|however|yet)\s+(?:if|then)', text, re.IGNORECASE))

    return multiple_answers or unnecessary_continuation


def detect_incoherent(text: str) -> bool:
    """
    Detect Incoherent/Garbled: output is unreadable or indicates crash.
    Heuristic: incomplete sentences, strange Unicode, repeated characters, etc.
    """
    if not text or len(text) < 10:
        return True

    # Check for excessive repeated characters
    repeated = re.findall(r'(.)\1{5,}', text)
    if repeated:
        return True

    # Check for incomplete sentences (many lines ending with "the" or "and")
    lines = text.split('\n')
    incomplete_endings = sum(1 for line in lines if line.rstrip().endswith(('and', 'the', 'or', 'is')))
    if incomplete_endings > len(lines) * 0.5:
        return True

    # Check for very few words
    word_count = len(text.split())
    if word_count < 5:
        return True

    return False


def classify_output(
    generated_text: str,
    expected_answer: str,
    question: str,
) -> Tuple[str, float]:
    """
    Classify output into one of 6 categories.
    Returns: (category, confidence_score)
    """

    # Handle errors/empty outputs
    if not generated_text or "error" in generated_text.lower():
        return "INCOHERENT", 1.0

    # Check for incoherence first (highest priority)
    if detect_incoherent(generated_text):
        return "INCOHERENT", 1.0

    # Check answer correctness
    is_correct = check_answer_correctness(generated_text, expected_answer)

    if not is_correct:
        # Wrong answer - could be Premise Hijacking, Confidence Snowballing, or Overcounting
        if detect_premise_hijacking(generated_text, question):
            return "PREMISE_HIJACKING", 0.7
        elif detect_confidence_snowballing(generated_text):
            return "CONFIDENCE_SNOWBALLING", 0.7
        elif detect_overcounting(generated_text, question):
            return "OVERCOUNTING", 0.7
        else:
            # Wrong answer but no specific pattern
            return "CONFIDENCE_SNOWBALLING", 0.5

    # Answer is correct
    if detect_shortcut_collapse(generated_text):
        return "SHORTCUT_COLLAPSE", 0.7
    else:
        return "NO_FAILURE", 0.9


def score_results(results_file: str) -> Dict:
    """
    Score all results in a results JSONL file.
    """
    scores_by_category = defaultdict(int)
    scores_by_source = defaultdict(lambda: defaultdict(int))
    all_scores = []

    with open(results_file, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue

            generated_text = result.get("generated_text", "")
            expected_answer = result.get("expected_answer", "")
            question = result.get("question", "")
            source = result.get("source", "unknown")

            # Classify
            category, confidence = classify_output(generated_text, expected_answer, question)

            # Record
            scores_by_category[category] += 1
            scores_by_source[source][category] += 1

            all_scores.append({
                "problem_id": result.get("problem_id"),
                "source": source,
                "category": category,
                "confidence": confidence,
                "is_correct": check_answer_correctness(generated_text, expected_answer),
            })

    return {
        "total_problems": len(all_scores),
        "by_category": dict(scores_by_category),
        "by_source": {source: dict(cats) for source, cats in scores_by_source.items()},
        "all_scores": all_scores,
    }


def generate_report(results_dir: str) -> Dict:
    """
    Score all three runs (auto, fp8, int8_per_token_head) and generate comparison.
    """
    dtypes = ["auto", "fp8", "int8_per_token_head"]
    all_results = {}

    print("\n" + "="*70)
    print("SCORING RESULTS")
    print("="*70 + "\n")

    for dtype in dtypes:
        results_file = Path(results_dir) / dtype / "outputs.jsonl"
        if not results_file.exists():
            print(f"⚠ Results not found for {dtype}: {results_file}")
            continue

        print(f"Scoring {dtype}...")
        scores = score_results(str(results_file))
        all_results[dtype] = scores

        # Save scores
        scores_file = results_file.parent / "scores.json"
        with open(scores_file, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"  ✓ Scores saved to {scores_file}\n")

    # Generate comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70 + "\n")

    # Overall accuracy (no failure)
    print("Overall Correctness (No Failure category):")
    print("-" * 50)
    for dtype in dtypes:
        if dtype not in all_results:
            continue
        total = all_results[dtype]["total_problems"]
        no_failure = all_results[dtype]["by_category"].get("NO_FAILURE", 0)
        accuracy = (no_failure / total * 100) if total > 0 else 0
        print(f"  {dtype:25s}: {no_failure:3d}/{total:3d} ({accuracy:5.1f}%)")

    # Breakdown by category
    print("\n\nFailure Mode Breakdown:")
    print("-" * 70)
    categories = ["NO_FAILURE", "SHORTCUT_COLLAPSE", "PREMISE_HIJACKING", "CONFIDENCE_SNOWBALLING", "OVERCOUNTING", "INCOHERENT"]

    for dtype in dtypes:
        if dtype not in all_results:
            continue
        print(f"\n{dtype.upper()}:")
        total = all_results[dtype]["total_problems"]
        for cat in categories:
            count = all_results[dtype]["by_category"].get(cat, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {cat:25s}: {count:3d}/{total:3d} ({pct:5.1f}%)")

    # Breakdown by source (AIME vs GSM8K)
    print("\n\nResults by Problem Source:")
    print("-" * 70)

    for source in ["aime", "gsm8k"]:
        print(f"\n{source.upper()}:")
        for dtype in dtypes:
            if dtype not in all_results or source not in all_results[dtype]["by_source"]:
                continue
            source_results = all_results[dtype]["by_source"].get(source, {})
            no_failure = source_results.get("NO_FAILURE", 0)
            total_source = sum(source_results.values())
            accuracy = (no_failure / total_source * 100) if total_source > 0 else 0
            print(f"  {dtype:25s}: {no_failure:3d}/{total_source:3d} ({accuracy:5.1f}%)")

    print("\n" + "="*70)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Score vLLM experiment results")
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory containing results subdirs (auto, fp8, int8_per_token_head)",
    )

    args = parser.parse_args()

    if not Path(args.results_dir).exists():
        print(f"✗ Results directory not found: {args.results_dir}")
        exit(1)

    generate_report(args.results_dir)


if __name__ == "__main__":
    main()
