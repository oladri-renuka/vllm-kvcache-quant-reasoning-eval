# vLLM KV-Cache Quantization Experiment Report

## Executive Summary

This report documents an experiment investigating whether the new `int8_per_token_head` KV-cache quantization in vLLM introduces silent reasoning failures (Shortcut Collapse) compared to baseline and FP8 quantization, similar to the NF4 weight quantization findings reported in [paper citation].

**Date:** [FILL: Date experiment was run]  
**Model:** [FILL: Model name used]  
**Experiment ID:** [FILL: Optional ID for tracking]

---

## Experiment Design

### Test Set
- **Total Problems:** 100
- **AIME 2025 (hard math reasoning):** 30 problems
- **GSM8K (grade-school math):** 70 problems
- **Data Source:** 
  - AIME: [FILL: HuggingFace dataset or source]
  - GSM8K: `openai/gsm8k` from HuggingFace Hub

### Configurations Tested

| Config | KV-Cache Dtype | Backend | Description |
|--------|----------------|---------|-------------|
| Baseline | `auto` | FLASHINFER | No compression (reference) |
| FP8 | `fp8` | FLASHINFER | FP8 quantization |
| Target | `int8_per_token_head` | FLASHINFER | Per-token per-head int8 quantization |

**Inference Settings:**
- Temperature: 0.7
- Top-p: 0.95
- Max tokens: 1024
- Seed: 42 (reproducible)
- Attention backend: FLASHINFER (to avoid Triton bug #49716)

### Failure Mode Taxonomy

Outputs are classified into six categories:

1. **No Failure:** Correct reasoning + correct answer
2. **Shortcut Collapse:** Correct answer but reasoning is incomplete, skipped, or unverifiable
3. **Premise Hijacking:** Model accepts a false assumption and reasons correctly from it
4. **Confidence Snowballing:** A single early error propagates through the solution
5. **Overcounting:** Correct intermediate answer, then continues unnecessarily
6. **Incoherent/Garbled:** Output is unreadable, incomplete, or indicates a crash

---

## Results

### Overall Accuracy (No Failure Category)

| Configuration | Correct | Total | Accuracy | 95% CI |
|---------------|---------|-------|----------|--------|
| Baseline (auto) | [FILL] | 100 | [FILL]% | [FILL] |
| FP8 | [FILL] | 100 | [FILL]% | [FILL] |
| int8_per_token_head | [FILL] | 100 | [FILL]% | [FILL] |

**Key Finding:** [FILL: Does int8_per_token_head match baseline? Does it degrade?]

### Failure Mode Breakdown

#### Baseline (auto)
| Failure Mode | Count | % |
|--------------|-------|---|
| No Failure | [FILL] | [FILL]% |
| Shortcut Collapse | [FILL] | [FILL]% |
| Premise Hijacking | [FILL] | [FILL]% |
| Confidence Snowballing | [FILL] | [FILL]% |
| Overcounting | [FILL] | [FILL]% |
| Incoherent | [FILL] | [FILL]% |

#### FP8
| Failure Mode | Count | % |
|--------------|-------|---|
| No Failure | [FILL] | [FILL]% |
| Shortcut Collapse | [FILL] | [FILL]% |
| Premise Hijacking | [FILL] | [FILL]% |
| Confidence Snowballing | [FILL] | [FILL]% |
| Overcounting | [FILL] | [FILL]% |
| Incoherent | [FILL] | [FILL]% |

#### int8_per_token_head
| Failure Mode | Count | % |
|--------------|-------|---|
| No Failure | [FILL] | [FILL]% |
| Shortcut Collapse | [FILL] | [FILL]% |
| Premise Hijacking | [FILL] | [FILL]% |
| Confidence Snowballing | [FILL] | [FILL]% |
| Overcounting | [FILL] | [FILL]% |
| Incoherent | [FILL] | [FILL]% |

### Results by Problem Difficulty

#### AIME (Hard Math Reasoning)

| Configuration | Correct | Total | Accuracy |
|---------------|---------|-------|----------|
| Baseline (auto) | [FILL] | 30 | [FILL]% |
| FP8 | [FILL] | 30 | [FILL]% |
| int8_per_token_head | [FILL] | 30 | [FILL]% |

**Shortcut Collapse on AIME:**
- Baseline: [FILL]
- FP8: [FILL]
- int8_per_token_head: [FILL]

#### GSM8K (Grade-School Math)

| Configuration | Correct | Total | Accuracy |
|---------------|---------|-------|----------|
| Baseline (auto) | [FILL] | 70 | [FILL]% |
| FP8 | [FILL] | 70 | [FILL]% |
| int8_per_token_head | [FILL] | 70 | [FILL]% |

**Shortcut Collapse on GSM8K:**
- Baseline: [FILL]
- FP8: [FILL]
- int8_per_token_head: [FILL]

### Performance Metrics

| Configuration | Avg Latency (s/problem) | Total GPU Memory (GB) | Peak Memory (GB) |
|---------------|------------------------|-----------------------|------------------|
| Baseline (auto) | [FILL] | [FILL] | [FILL] |
| FP8 | [FILL] | [FILL] | [FILL] |
| int8_per_token_head | [FILL] | [FILL] | [FILL] |

---

## Analysis & Interpretation

### Main Question
**Does `int8_per_token_head` KV-cache quantization cause Shortcut Collapse like NF4 weight quantization?**

[FILL: Detailed interpretation of results. Address the main question directly.]

**Finding:**  
[FILL: Concise summary]

### Comparison with Prior Work
Based on the NF4 weight quantization paper:
- NF4 caused Shortcut Collapse in [X]% of problems
- int8_per_token_head causes Shortcut Collapse in [Y]% of problems
- [FILL: Comparison and analysis]

### Shortcut Collapse Deep Dive

**Shortcut Collapse Rate by Config:**
- Baseline: [FILL]% of failures are Shortcut Collapse
- FP8: [FILL]% of failures are Shortcut Collapse
- int8_per_token_head: [FILL]% of failures are Shortcut Collapse

**Key Observation:**  
[FILL: Is Shortcut Collapse elevated in int8_per_token_head? By how much?]

### Other Failure Modes

**Premise Hijacking, Confidence Snowballing, Overcounting:**
[FILL: Do these failure modes change across configurations?]

### Statistical Significance

[FILL: If any accuracy differences exist between configs, are they statistically significant? Use binomial test or similar.]

---

## Reproducibility

### System Configuration
- **GPU:** [FILL: Model and VRAM]
- **CUDA Version:** [FILL]
- **PyTorch Version:** [FILL]
- **vLLM Version:** [FILL]
- **Transformers Version:** [FILL]

### Running the Experiment
All code and data are available in this directory. To reproduce:

```bash
# 1. Setup
make setup

# 2. Prepare test data
make prepare

# 3. Run experiments
make run-all

# 4. Score results
make score

# 5. Manual review
make sanity-check
```

**Exact Commands Used:**
```bash
# Baseline
python scripts/run_experiment.py --kv_cache_dtype auto

# FP8
python scripts/run_experiment.py --kv_cache_dtype fp8

# int8_per_token_head
python scripts/run_experiment.py --kv_cache_dtype int8_per_token_head
```

### Data & Outputs
- Test set: `data/testset.jsonl` (100 problems, 30 AIME + 70 GSM8K)
- Raw outputs: `results/{dtype}/outputs.jsonl` (complete generation text per problem)
- Scores: `results/{dtype}/scores.json` (category classification + statistics)

**All outputs are publicly available in this repository.**

---

## Conclusions

### Key Findings

1. **int8_per_token_head accuracy:** [FILL: Matches baseline / degrades by X%]
2. **Shortcut Collapse rate:** [FILL: Elevated / matches / lower than baseline]
3. **Failure mode profile:** [FILL: Different from or similar to baseline?]

### Implications for vLLM

[FILL: What do these results mean for users considering int8_per_token_head?]

### Recommendations

- For production use: [FILL: Recommend int8_per_token_head or caution against?]
- For further research: [FILL: Any follow-up experiments suggested?]

---

## Appendix: Sample Failures

### Shortcut Collapse Example (int8_per_token_head)
[FILL: Paste one or two example outputs showing Shortcut Collapse]

### Confidence Snowballing Example
[FILL: Example output if present]

---

## Revision History

| Date | Author | Notes |
|------|--------|-------|
| [FILL] | [FILL] | Initial experiment run |

---

**Generated:** [FILL: Script run date]  
**Reviewed:** [FILL: Manual review date]  
**For:** vLLM Issue #33480 — int8_per_token_head KV-cache quantization safety analysis
