#!/usr/bin/env python3
"""
Calibrate FP8 KV-cache scales using llm-compressor.
Based on vLLM docs example.
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# Config
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NUM_CALIB_SAMPLES = 20  # Use first 20 problems for calibration
MAX_SEQ_LEN = 2048
STRATEGY = "tensor"  # or "attn_head" (requires Flash Attention)
OUTPUT_DIR = "results/qwen25_fp8_calibrated"

def load_calib_data():
    """Load first N problems from testset for calibration."""
    testset_path = "data/testset.jsonl"
    problems = []

    with open(testset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= NUM_CALIB_SAMPLES:
                break
            if line.strip():
                problems.append(json.loads(line))

    return problems

def create_prompt(problem):
    """Create chat-formatted prompt."""
    prompt = f"""You are a helpful math assistant. Solve the following problem step by step, showing your reasoning.

Problem: {problem['question']}

Please provide your step-by-step solution and final answer."""
    return prompt

def main():
    print("\n" + "="*70)
    print("FP8 KV-Cache Calibration")
    print("="*70 + "\n")

    # Load model and tokenizer
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load calibration data
    print(f"\nLoading calibration dataset ({NUM_CALIB_SAMPLES} problems)...")
    problems = load_calib_data()

    # Convert to text for tokenization
    texts = [create_prompt(p) for p in problems]

    # Tokenize
    print("Tokenizing calibration data...")
    tokenized = tokenizer(
        texts,
        padding=True,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )

    # Create HF dataset format for llm-compressor
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    })

    # Define quantization recipe
    print(f"\nSetting up {STRATEGY} FP8 quantization recipe...")
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=STRATEGY)
    recipe = QuantizationModifier(
        kv_cache_scheme=fp8_args,  # Quantize KV cache
    )

    # Run calibration
    print("\nRunning one-shot calibration...")
    print("(This may take a few minutes...)\n")
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
    )

    # Save quantized model
    print(f"\nSaving calibrated model to {OUTPUT_DIR}...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✓ Calibration complete!")
    print(f"✓ Quantized model saved to: {OUTPUT_DIR}")
    print(f"\nTo use this model with vLLM:")
    print(f"  llm = LLM(model='{OUTPUT_DIR}', kv_cache_dtype='fp8')")

if __name__ == "__main__":
    main()
