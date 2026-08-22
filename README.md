# vLLM KV-Cache Quantization Safety Experiment

**Testing whether `int8_per_token_head` KV-cache quantization causes silent reasoning failures (Shortcut Collapse) in vLLM.**

This is a **real, reproducible experiment** with:
- ✅ Actual vLLM inference (no mocks)
- ✅ Real test data (100 math problems: 30 AIME + 70 GSM8K)
- ✅ Three genuine configurations (baseline, FP8, int8_per_token_head)
- ✅ Automated failure classification (6-category taxonomy)
- ✅ Manual sanity checks

**Ref:** vLLM Issue #33480

---

## Quick Start

### 1. Setup (one time)
```bash
make setup
```
This installs vLLM from source + Python dependencies.

### 2. Prepare Test Data
```bash
make prepare
```
Downloads 30 real AIME 2025 + 70 real GSM8K problems.

### 3. Run Experiments (takes 2-4 hours on A100)
```bash
make run-all
```
Or individual runs:
```bash
make run-baseline   # auto (reference)
make run-fp8        # FP8 quantization
make run-int8       # int8_per_token_head (target)
```

### 4. Score Results
```bash
make score
```
Applies the 6-category taxonomy to all outputs.

### 5. Manual Review
```bash
make sanity-check
```
Spot-checks 5 random outputs per dtype for verification.

### 6. Generate Report
```bash
make report
```
Shows comparison table and summary.

---

## For Testing with Fewer Problems

Run a quick test with, e.g., 5 problems per dtype:

```bash
make run-baseline NUM_PROBLEMS=5
make run-fp8 NUM_PROBLEMS=5
make run-int8 NUM_PROBLEMS=5
make score
```

**Note:** With only 5 problems, results won't be statistically meaningful, but you can verify the pipeline works.

---

## Directory Structure

```
vllm_kvcache_experiment/
├── scripts/
│   ├── setup.sh                 # Install dependencies
│   ├── prepare_testset.py        # Download & prepare test data
│   ├── run_experiment.py         # Run vLLM inference
│   ├── score_outputs.py          # Apply taxonomy classification
│   └── sanity_check.py           # Manual review samples
├── results/
│   ├── auto/
│   │   ├── outputs.jsonl         # Full outputs from baseline run
│   │   └── scores.json           # Classification scores
│   ├── fp8/
│   │   ├── outputs.jsonl
│   │   └── scores.json
│   └── int8_per_token_head/
│       ├── outputs.jsonl
│       └── scores.json
├── Makefile                      # Convenience targets
├── requirements.txt              # Python dependencies
├── report.md                     # Results summary
└── README.md                     # This file
```

---

## Test Set Format

Test data is generated at runtime by `prepare_testset.py` into `data/testset.jsonl`. Each line is a JSON object:
```json
{
  "question": "...",
  "answer": "...",
  "source": "aime" or "gsm8k"
}
```

---

## Output Format

`results/{dtype}/outputs.jsonl`: Each line contains:
```json
{
  "problem_id": 1,
  "source": "aime",
  "question": "...",
  "expected_answer": "...",
  "generated_text": "...",
  "token_count": 150,
  "latency_seconds": 2.5,
  "memory_used_gb": 0.3,
  "timestamp": "2025-08-21T12:34:56.789012"
}
```

---

## Scoring Taxonomy

Outputs are classified into six failure modes:

| Category | Description |
|----------|-------------|
| **No Failure** | Correct reasoning + correct answer |
| **Hollow Convergence** | Correct answer, but reasoning is incomplete/skipped/vague |
| **Shortcut Collapse** | Incorrect answer due to unjustified logical leaps |
| **Premise Hijacking** | Accepts a false assumption, reasons correctly from it |
| **Confidence Snowballing** | A single early error propagates through the solution |
| **Overcounting** | Correct intermediate answer, then continues unnecessarily |

**Scoring Logic:** `score_outputs.py` implements LLM-as-judge classification. See the code for detailed prompts.

---

## Configuration Details

### Model
Default: `Qwen/Qwen2.5-7B-Instruct` (smaller, fits on most GPUs)  
Alternative: `meta-llama/Llama-3.1-8B-Instruct` (change in `run_experiment.py` or pass `--model`)

### Inference Settings
- Temperature: 0.7 (some randomness for interesting outputs)
- Top-p: 0.95
- Max tokens: 1024 (allow full reasoning chains)
- Seed: 42 (reproducible across runs)
- Attention backend: FLASHINFER (avoids Triton bug #49716)

### Important Notes
- Use **FLASHINFER** backend to avoid Triton quantization bug
- Choose a non-hybrid architecture to avoid Gemma bug
- KV-cache quantization only affects inference, not training
- Results are reproducible if you use the same seed and model

---

## Running on GPU Instances

This experiment is designed to run on RunPod or similar cloud GPU instances:

```bash
# On a RunPod A100 instance:
git clone <this-repo>
cd vllm_kvcache_experiment
make setup
make prepare
make run-all          # ~3-4 hours
make score
make sanity-check
# Fill in report.md with results
```

---

## Interpreting Results

### Primary Question
**Does `int8_per_token_head` cause Shortcut Collapse more than baseline or FP8?**

Expected interpretations:
- **✓ Good:** Accuracy matches baseline (within statistical noise), Shortcut Collapse rate is similar
- **⚠ Concerning:** Accuracy degrades OR Shortcut Collapse rate is significantly elevated
- **🚨 Red Flag:** Accuracy drops >5% OR Shortcut Collapse becomes dominant failure mode

### Sanity Checks
1. **Manual review:** Look at 5-10 outputs per dtype. Do the predicted categories match reality?
2. **Spot-check correctness:** Pick 2-3 correct outputs and 2-3 incorrect outputs. Does the reasoning match the category?
3. **Compare with baseline:** Are baseline (auto) results reasonable? (Typically 60-80% correct on this mixed test set)

---

## Troubleshooting

### Out of Memory
- Reduce `gpu_memory_utilization` in `run_experiment.py` (default 0.9)
- Use a smaller model (e.g., Qwen2.5-3B instead of 7B)
- Reduce `max_tokens` in `SamplingParams`

### vLLM Installation Fails
Make sure you have CUDA 11.8+ and that PyTorch installed correctly:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Scores not generated
Make sure `results/{dtype}/outputs.jsonl` exists and has content:
```bash
wc -l results/auto/outputs.jsonl
head results/auto/outputs.jsonl
```

### Model download fails
HuggingFace token may be required:
```bash
huggingface-cli login
```

---

## Contributing Back to vLLM

Once results are ready:

1. Fill in `report.md` with your findings
2. Post the results to [vLLM Issue #33480](https://github.com/vllm-project/vllm/issues/33480)
3. Include:
   - Accuracy table (auto vs fp8 vs int8_per_token_head)
   - Shortcut Collapse rates
   - Your GPU model and CUDA version
   - Any interesting failure examples

---

## Citation

If you use this experiment framework, please cite:

```bibtex
@software{vllm_kvcache_experiment,
  title = {vLLM KV-Cache Quantization Safety Experiment},
  year = {2025},
  note = {GitHub: Open VLM / vLLM Issue \#33480}
}
```

---

## Questions?

- See `report.md` for detailed results template
- Check `scripts/` for implementation details
- Review logs in `results/{dtype}/` for diagnostics
