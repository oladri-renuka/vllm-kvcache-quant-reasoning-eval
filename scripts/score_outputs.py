#!/usr/bin/env python3
"""
Score outputs using the 6-category taxonomy from:
"Silent Failures in Quantized LLM Reasoning" (Oladri et al., arXiv:2607.09999)

Uses LLM-as-judge (OpenRouter API) or fallback to heuristics if API unavailable.

Categories:
1. NO_FAILURE: Correct answer with sound reasoning
2. HOLLOW_CONVERGENCE: Correct answer but reasoning incomplete/skipped
3. PREMISE_HIJACKING: False assumption, correct logic
4. SHORTCUT_COLLAPSE: Bypassed steps, unjustified leaps
5. OVERCOUNTING: Correct intermediate, continues unnecessarily
6. CONFIDENCE_SNOWBALLING: Early error propagates through reasoning
"""

import json
import argparse
import re
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge Prompts (from Oladri et al. paper)
# ────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_PASS1 = """
### Role
Expert Logic Auditor specializing in LLM Error Taxonomy.

### Task
Analyze the provided Reasoning Chain that led to a WRONG answer. Identify the exact moment the logic failed and classify the error into ONE category.

### Error Taxonomy
1. PREMISE_HIJACKING: The model accepted a false, paradoxical, or misleading assumption in the question and reasoned correctly from that flawed foundation. The logic is internally consistent but built on a wrong premise.
2. SHORTCUT_COLLAPSE: The model bypassed necessary logical steps, made unjustified leaps of inference, or hallucinated a connection to reach the final answer without showing required intermediate work.
3. OVERCOUNTING: The model correctly reached an intermediate milestone but continued reasoning past it, adding redundant or contradictory steps that invalidated its own correct progress.
4. CONFIDENCE_SNOWBALLING: A single small error in an early reasoning step propagated silently through otherwise perfect subsequent reasoning, producing a coherent but incorrect final answer.

### Inputs
- Question: {question}
- Ground Truth: {ground_truth}
- Model Reasoning: {cot_chain}
- Model Answer: {model_answer}

### Output Constraint
Return ONLY a valid JSON object. No markdown, no backticks, no explanation outside the JSON.
{{"category": "ONE OF: PREMISE_HIJACKING, SHORTCUT_COLLAPSE, OVERCOUNTING, CONFIDENCE_SNOWBALLING", "justification": "One sentence identifying the specific point where the reasoning failed and why it fits the chosen category."}}
"""

JUDGE_PROMPT_PASS2 = """
### Role
Expert Logic Auditor.

### Task
The model reached the CORRECT final answer. Determine whether this represents genuine valid reasoning or hollow convergence.

### Categories
1. NO_FAILURE: The reasoning is robust. Every intermediate step is logically and mathematically derived from the previous one. All steps are visible and sound.
2. HOLLOW_CONVERGENCE: The model arrived at the correct answer but through incomplete, skipped, or hollow reasoning steps. The answer is right but the reasoning chain would not hold up to scrutiny.

### Inputs
- Question: {question}
- Ground Truth: {ground_truth}
- Model Reasoning: {cot_chain}

### Output Constraint
Return ONLY a valid JSON object. No markdown, no backticks, no explanation outside the JSON.
{{"category": "ONE OF: NO_FAILURE, HOLLOW_CONVERGENCE", "justification": "One sentence explaining whether the reasoning is genuinely valid or hollow."}}
"""

VALID_CATS = {
    'PREMISE_HIJACKING',
    'SHORTCUT_COLLAPSE',
    'OVERCOUNTING',
    'CONFIDENCE_SNOWBALLING',
    'NO_FAILURE',
    'HOLLOW_CONVERGENCE',
}

JUDGE_MODELS = [
    'meta-llama/llama-3.3-70b-instruct:free',
    'openai/gpt-oss-120b:free',
    'nvidia/nemotron-3-super-120b-a12b:free',
]


def normalize_answer(answer_str: str) -> str:
    """Normalize answer for comparison."""
    answer = answer_str.strip().lower()
    answer = re.sub(r'\s+', ' ', answer)
    return answer


def check_answer_correctness(generated_text: str, expected_answer: str) -> bool:
    """Check if the generated output contains the expected answer."""
    expected_norm = normalize_answer(expected_answer)
    generated_norm = normalize_answer(generated_text)

    if expected_norm in generated_norm:
        return True

    expected_nums = re.findall(r'[\d.]+', expected_norm)
    generated_nums = re.findall(r'[\d.]+', generated_norm)

    if expected_nums and generated_nums:
        for exp in expected_nums:
            if exp in generated_nums:
                return True

    return False


# ────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge Classification
# ────────────────────────────────────────────────────────────────────────────

def classify_with_llm_judge(
    question: str,
    ground_truth: str,
    cot_chain: str,
    model_answer: str,
    is_correct: bool,
) -> Tuple[str, str]:
    """
    Classify output using LLM-as-judge (OpenRouter API).
    Returns: (category, justification)
    """
    if not HAS_OPENAI:
        print("  [WARN] openai not installed, falling back to heuristics")
        return None, ""

    openrouter_key = os.getenv('OPENROUTER_KEY')
    if not openrouter_key:
        print("  [WARN] OPENROUTER_KEY not set, falling back to heuristics")
        return None, ""

    client = openai.OpenAI(
        api_key=openrouter_key,
        base_url='https://openrouter.ai/api/v1'
    )

    if is_correct:
        prompt = JUDGE_PROMPT_PASS2.format(
            question=question,
            ground_truth=ground_truth,
            cot_chain=cot_chain[:3000],
        )
    else:
        prompt = JUDGE_PROMPT_PASS1.format(
            question=question,
            ground_truth=ground_truth,
            cot_chain=cot_chain[:3000],
            model_answer=model_answer,
        )

    for model in JUDGE_MODELS:
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=200,
                    messages=[{'role': 'user', 'content': prompt}],
                    extra_headers={
                        'HTTP-Referer': 'https://umd.edu',
                        'X-Title': 'vLLM KV-Cache Safety Research'
                    }
                )
                raw = resp.choices[0].message.content.strip()
                if '```' in raw:
                    parts = raw.split('```')
                    for part in parts:
                        part = part.lstrip('json').strip()
                        if part.startswith('{'):
                            raw = part
                            break
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not json_match:
                    time.sleep(2)
                    continue
                data = json.loads(json_match.group())
                c = data.get('category', 'UNKNOWN').upper().strip()
                cat = c if c in VALID_CATS else 'UNKNOWN'
                just = data.get('justification', '')
                if cat != 'UNKNOWN':
                    return cat, just
            except (json.JSONDecodeError, Exception) as e:
                time.sleep(2)
        print(f'  [FALLBACK] {model} failed, trying next...')

    return None, ""


# ────────────────────────────────────────────────────────────────────────────
# Fallback Heuristic Classification (if API unavailable)
# ────────────────────────────────────────────────────────────────────────────

def classify_with_heuristics(
    generated_text: str,
    expected_answer: str,
    question: str,
    is_correct: bool,
) -> Tuple[str, str]:
    """
    Fallback heuristic classification when LLM-as-judge is unavailable.
    """
    if not generated_text:
        return "UNKNOWN", ""

    # Check answer correctness
    if not is_correct:
        if len(generated_text) > 200 and any(word in generated_text.lower()
                                            for word in ['assume', 'suppose', 'let']):
            return "PREMISE_HIJACKING", "Possible flawed assumption"
        elif len(re.findall(r'[\+\-\*/]', generated_text)) > 10:
            return "CONFIDENCE_SNOWBALLING", "Multiple calculations with possible error cascade"
        elif len(re.findall(r'answer', generated_text, re.IGNORECASE)) > 1:
            return "OVERCOUNTING", "Multiple answers detected"
        else:
            return "CONFIDENCE_SNOWBALLING", "Wrong answer with unclear failure mode"

    # Answer is correct
    if len(generated_text) < 100:
        return "HOLLOW_CONVERGENCE", "Very short output, likely incomplete reasoning"
    elif not any(re.search(pattern, generated_text, re.IGNORECASE)
                for pattern in [r'step', r'therefore', r'so', r'this means']):
        return "HOLLOW_CONVERGENCE", "No reasoning markers found"
    else:
        return "NO_FAILURE", "Sound reasoning detected"


def classify_output(
    generated_text: str,
    expected_answer: str,
    question: str,
    ground_truth: str = None,
    model_answer: str = None,
) -> Tuple[str, str]:
    """
    Classify output using LLM-as-judge, fallback to heuristics.
    Returns: (category, explanation)
    """
    # Handle empty/error outputs
    if not generated_text or "error" in generated_text.lower():
        return "UNKNOWN", "Empty or error output"

    is_correct = check_answer_correctness(generated_text, expected_answer)

    # Try LLM-as-judge first
    if HAS_OPENAI and ground_truth and model_answer is not None:
        cat, expl = classify_with_llm_judge(question, ground_truth, generated_text, model_answer, is_correct)
        if cat and cat in VALID_CATS:
            return cat, expl

    # Fallback to heuristics
    return classify_with_heuristics(generated_text, expected_answer, question, is_correct)


def score_results(results_file: str) -> Dict:
    """Score all results in a results JSONL file using LLM-as-judge or heuristics."""
    scores_by_category = defaultdict(int)
    scores_by_source = defaultdict(lambda: defaultdict(int))
    all_scores = []

    with open(results_file, 'r') as f:
        lines = [l for l in f if l.strip()]
        total_lines = len(lines)

        for idx, line in enumerate(lines):
            try:
                result = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [PARSE ERROR] Line {idx}: {e}")
                continue

            generated_text = result.get("generated_text", "")
            expected_answer = result.get("expected_answer", "")
            question = result.get("question", "")
            source = result.get("source", "unknown")
            problem_id = result.get("problem_id", idx)

            # Classify
            category, explanation = classify_output(
                generated_text, expected_answer, question,
                ground_truth=expected_answer,
                model_answer=generated_text,
            )

            is_correct = check_answer_correctness(generated_text, expected_answer)

            # Record
            scores_by_category[category] += 1
            scores_by_source[source][category] += 1

            all_scores.append({
                "problem_id": problem_id,
                "source": source,
                "category": category,
                "explanation": explanation,
                "is_correct": is_correct,
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == total_lines:
                print(f"    {idx+1}/{total_lines} scored | last: {category}")

    return {
        "total_problems": len(all_scores),
        "by_category": dict(scores_by_category),
        "by_source": {source: dict(cats) for source, cats in scores_by_source.items()},
        "all_scores": all_scores,
    }


def generate_report(results_dir: str) -> Dict:
    """Score all three runs and generate comparison using Oladri et al. taxonomy."""
    dtypes = ["auto", "fp8", "int8_per_token_head"]
    all_results = {}

    print("\n" + "="*80)
    print("SCORING RESULTS (Oladri et al. taxonomy via LLM-as-judge)")
    print("="*80 + "\n")

    for dtype in dtypes:
        results_file = Path(results_dir) / dtype / "outputs.jsonl"
        if not results_file.exists():
            print(f"⚠ Results not found for {dtype}: {results_file}")
            continue

        print(f"Scoring {dtype}...")
        scores = score_results(str(results_file))
        all_results[dtype] = scores

        scores_file = results_file.parent / "scores.json"
        with open(scores_file, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"  ✓ Saved to {scores_file}\n")

    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80 + "\n")

    # Pass rate (NO_FAILURE + HOLLOW_CONVERGENCE = correct answers)
    print("Correct Answer Rate:")
    print("-" * 60)
    for dtype in dtypes:
        if dtype not in all_results:
            continue
        total = all_results[dtype]["total_problems"]
        correct = sum(s["is_correct"] for s in all_results[dtype]["all_scores"])
        rate = (correct / total * 100) if total > 0 else 0
        print(f"  {dtype:25s}: {correct:3d}/{total:3d} ({rate:5.1f}%)")

    # Hollow Convergence (the key metric from the paper!)
    print("\n\nHollow Convergence Rate (Correct Answer, Incomplete Reasoning):")
    print("-" * 60)
    for dtype in dtypes:
        if dtype not in all_results:
            continue
        total = all_results[dtype]["total_problems"]
        hc = all_results[dtype]["by_category"].get("HOLLOW_CONVERGENCE", 0)
        rate = (hc / total * 100) if total > 0 else 0
        print(f"  {dtype:25s}: {hc:3d}/{total:3d} ({rate:5.1f}%)")

    # Full breakdown
    print("\n\nFull Failure Mode Breakdown:")
    print("-" * 80)
    categories = ["NO_FAILURE", "HOLLOW_CONVERGENCE", "PREMISE_HIJACKING",
                  "SHORTCUT_COLLAPSE", "OVERCOUNTING", "CONFIDENCE_SNOWBALLING", "UNKNOWN"]

    for dtype in dtypes:
        if dtype not in all_results:
            continue
        print(f"\n{dtype.upper()}:")
        total = all_results[dtype]["total_problems"]
        for cat in categories:
            count = all_results[dtype]["by_category"].get(cat, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {cat:30s}: {count:3d}/{total:3d} ({pct:5.1f}%)")

    # By source
    print("\n\nResults by Problem Source (AIME vs GSM8K):")
    print("-" * 80)

    for source in ["aime", "gsm8k"]:
        print(f"\n{source.upper()}:")
        for dtype in dtypes:
            if dtype not in all_results or source not in all_results[dtype]["by_source"]:
                continue
            source_results = all_results[dtype]["by_source"].get(source, {})
            total_source = sum(source_results.values())
            correct_src = sum(1 for s in all_results[dtype]["all_scores"]
                            if s["source"] == source and s["is_correct"])
            accuracy = (correct_src / total_source * 100) if total_source > 0 else 0
            hc_src = source_results.get("HOLLOW_CONVERGENCE", 0)
            print(f"  {dtype:25s}: {correct_src:3d}/{total_source:3d} ({accuracy:5.1f}%) | HC: {hc_src:3d}")

    print("\n" + "="*80)

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
