#!/usr/bin/env python3
"""
Test inference with calibrated FP8 model.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import time

from vllm import LLM, SamplingParams
from tqdm import tqdm

def load_testset(testset_path="data/testset.jsonl"):
    """Load test set from JSONL file."""
    problems = []
    with open(testset_path, 'r') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems

def create_prompt(question):
    """Create a chat-formatted prompt with CoT instruction."""
    prompt = f"""You are a helpful math assistant. Solve the following problem step by step, showing your reasoning.

Problem: {question}

Please provide your step-by-step solution and final answer."""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Test calibrated FP8 model")
    parser.add_argument(
        "--model-path",
        default="results/qwen25_fp8_calibrated",
        help="Path to calibrated model",
    )
    parser.add_argument(
        "--testset-path",
        default="data/testset.jsonl",
        help="Path to test set",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory",
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Testing Calibrated FP8 Model")
    print("="*70)

    # Load test set
    problems = load_testset(args.testset_path)
    print(f"\nLoaded {len(problems)} problems")

    # Initialize vLLM with calibrated model
    print(f"\nInitializing vLLM with calibrated model: {args.model_path}")
    try:
        llm = LLM(
            model=args.model_path,
            kv_cache_dtype="fp8",  # Use FP8 with calibrated scales
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            attention_backend="FLASH_ATTN",
        )
        print("✓ vLLM initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        raise

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        seed=42,
    )

    # Create output directory
    output_dir = Path(args.output_dir) / "fp8_calibrated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "outputs.jsonl"

    # Run inference
    print(f"\nRunning inference on {len(problems)} problems...")
    print(f"Output: {output_file}\n")

    with open(output_file, 'w') as f:
        for idx, problem in enumerate(tqdm(problems, desc="Inference")):
            start_time = time.time()

            prompt = create_prompt(problem["question"])
            try:
                outputs = llm.generate(prompt, sampling_params)
                generated_text = outputs[0].outputs[0].text
            except Exception as e:
                generated_text = f"[ERROR] {e}"

            latency = time.time() - start_time

            result = {
                "problem_id": idx,
                "source": problem.get("source", "unknown"),
                "question": problem["question"],
                "expected_answer": problem["answer"],
                "generated_text": generated_text,
                "token_count": len(generated_text.split()),
                "latency_seconds": latency,
                "timestamp": datetime.now().isoformat(),
            }

            f.write(json.dumps(result) + '\n')

    print(f"\n✓ Inference complete!")
    print(f"✓ Results saved to: {output_file}")
    print(f"\nNext: Run scoring to classify outputs")
    print(f"  python scripts/score_outputs.py --results_dir results")

if __name__ == "__main__":
    main()
