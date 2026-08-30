#!/usr/bin/env python3
"""
Run actual vLLM inference with different KV-cache quantization settings.
This script loads a real model, processes real test data, and captures real outputs.
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime
import time
import psutil
import torch

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


def get_system_info():
    """Capture system and GPU info."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
    }
    return info


def create_prompt(question, source="unknown"):
    """Create a chat-formatted prompt with CoT instruction."""
    # Use Llama 3.1 / Qwen 2.5 chat template
    prompt = f"""You are a helpful math assistant. Solve the following problem step by step, showing your reasoning.

Problem: {question}

Please provide your step-by-step solution and final answer."""
    return prompt


def run_experiment(
    kv_cache_dtype,
    model_name="Qwen/Qwen2.5-7B-Instruct",  # Smaller by default; change to meta-llama/Llama-3.1-8B-Instruct if you have GPU space
    testset_path="data/testset.jsonl",
    output_dir="results",
    num_problems=None,  # Use all if None
    calibrate=True,  # Enable KV-cache scale calibration for FP8
):
    """
    Run experiment with specified KV-cache dtype.
    """

    print("\n" + "="*70)
    print(f"vLLM Experiment: KV-Cache dtype = {kv_cache_dtype}")
    print("="*70)

    # Setup
    problems = load_testset(testset_path)
    if num_problems:
        problems = problems[:num_problems]

    dtype_dir = Path(output_dir) / kv_cache_dtype
    dtype_dir.mkdir(parents=True, exist_ok=True)
    output_file = dtype_dir / "outputs.jsonl"

    print(f"\nLoaded {len(problems)} problems from {testset_path}")
    print(f"Model: {model_name}")
    print(f"Output dir: {dtype_dir}")

    # System info
    system_info = get_system_info()
    print(f"\nSystem: {system_info['gpu_name']} (GPU: {system_info['gpu_available']})")

    # Initialize vLLM
    print(f"\nInitializing vLLM with kv_cache_dtype={kv_cache_dtype}...")
    print("(This may take a few minutes to load the model...)")

    try:
        # Build LLM kwargs
        llm_kwargs = {
            "model": model_name,
            "kv_cache_dtype": kv_cache_dtype,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
        }

        # Select attention backend based on KV-cache dtype
        if kv_cache_dtype == "int8_per_token_head":
            # Let vLLM auto-select backend for int8_per_token_head (don't force FLASH_ATTN)
            pass
        else:
            llm_kwargs["attention_backend"] = "FLASHINFER"  # Avoid Triton bug #49716 for other dtypes

        # Enable automatic KV-cache scale calibration for FP8 (if requested)
        if kv_cache_dtype == "fp8":
            llm_kwargs["calculate_kv_scales"] = calibrate
            if calibrate:
                print(f"  Enabling automatic KV-cache scale calibration (calculate_kv_scales=True)")
            else:
                print(f"  Using default FP8 scales (calculate_kv_scales=False)")

        llm = LLM(**llm_kwargs)
        print("✓ vLLM initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize vLLM: {e}")
        print("Make sure the model is available and you have enough GPU memory.")
        raise

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,  # Allow full reasoning chains
        seed=42,  # For reproducibility
    )

    # Track memory and timing
    process = psutil.Process()
    results = {
        "kv_cache_dtype": kv_cache_dtype,
        "model": model_name,
        "system_info": system_info,
        "sampling_params": {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_tokens,
        },
        "problems": []
    }

    # Run inference
    print(f"\nRunning inference on {len(problems)} problems...")
    print("(Outputs are being saved to {})".format(output_file))

    for idx, problem in enumerate(tqdm(problems, desc="Inference")):
        problem_id = idx + 1

        # Create prompt
        prompt = create_prompt(problem["question"], problem["source"])

        # Get baseline memory before inference
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        mem_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Run inference
        start_time = time.time()
        try:
            outputs = llm.generate([prompt], sampling_params=sampling_params)
            elapsed = time.time() - start_time
            generated_text = outputs[0].outputs[0].text

            # Track memory
            mem_after = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            mem_used = mem_after - mem_before

            # Token count (approximate from output length)
            token_count = len(generated_text.split())  # Rough estimate

            result = {
                "problem_id": problem_id,
                "source": problem["source"],
                "question": problem["question"],
                "expected_answer": problem["answer"],
                "generated_text": generated_text,
                "token_count": token_count,
                "latency_seconds": elapsed,
                "memory_used_gb": mem_used,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"\n✗ Error on problem {problem_id}: {e}")
            result = {
                "problem_id": problem_id,
                "source": problem["source"],
                "question": problem["question"],
                "expected_answer": problem["answer"],
                "generated_text": "",
                "error": str(e),
                "token_count": 0,
                "latency_seconds": 0,
                "memory_used_gb": 0,
                "timestamp": datetime.now().isoformat(),
            }

        results["problems"].append(result)

    # Save results
    with open(output_file, 'w') as f:
        for problem_result in results["problems"]:
            f.write(json.dumps(problem_result) + '\n')

    # Summary stats
    successful = sum(1 for p in results["problems"] if "error" not in p)
    failed = len(results["problems"]) - successful
    avg_latency = sum(p.get("latency_seconds", 0) for p in results["problems"]) / len(results["problems"])
    total_memory = sum(p.get("memory_used_gb", 0) for p in results["problems"])

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_file}")
    print(f"Successful: {successful}/{len(results['problems'])}")
    print(f"Failed: {failed}/{len(results['problems'])}")
    print(f"Avg latency: {avg_latency:.2f}s per problem")
    print(f"Total GPU memory used: {total_memory:.2f}GB")
    print("="*70)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM experiment with different KV-cache dtypes"
    )
    parser.add_argument(
        "--kv_cache_dtype",
        choices=["auto", "fp8", "int8_per_token_head"],
        default="auto",
        help="KV-cache data type",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name from HuggingFace Hub",
    )
    parser.add_argument(
        "--testset",
        default="data/testset.jsonl",
        help="Path to test set JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Number of problems to run (for testing). If None, use all.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=True,
        help="Enable automatic KV-cache scale calibration for FP8 (default: True)",
    )
    parser.add_argument(
        "--no-calibrate",
        dest="calibrate",
        action="store_false",
        help="Disable automatic KV-cache scale calibration for FP8 (use default scale=1.0)",
    )

    args = parser.parse_args()

    # Validate testset exists
    if not Path(args.testset).exists():
        print(f"✗ Test set not found: {args.testset}")
        print("Run: python scripts/prepare_testset.py")
        exit(1)

    # Run experiment
    run_experiment(
        kv_cache_dtype=args.kv_cache_dtype,
        model_name=args.model,
        testset_path=args.testset,
        output_dir=args.output_dir,
        num_problems=args.num_problems,
        calibrate=args.calibrate,
    )


if __name__ == "__main__":
    main()
