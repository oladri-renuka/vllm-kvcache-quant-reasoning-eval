# FP8 vs int8_per_token_head KV-Cache Quantization: Reasoning-Quality Evaluation

## Context

This repo answers an open question from [vllm-project/vllm#33480](https://github.com/vllm-project/vllm/issues/33480#issue-comment): does `int8_per_token_head` KV-cache quantization behave like FP8, and does either introduce *silent* reasoning-quality degradation — not just accuracy loss, but a change in *how* the model fails?

The taxonomy used to classify failures is from our prior work, [Silent Failures in Quantized LLM Reasoning](https://arxiv.org/abs/2607.09999) (Oladri et al., ICTAI 2026, submitted), which found that NF4 **weight** quantization increases Shortcut Collapse from 44% to 78% of wrong-answer failures, even when raw accuracy looks nearly unaffected. This experiment asks whether a similar silent pattern shows up under **KV-cache** quantization instead.

| Category | Description |
|---|---|
| **No Failure** | Correct reasoning + correct answer |
| **Hollow Convergence** | Correct answer, but reasoning is incomplete/skipped/vague |
| **Shortcut Collapse** | Incorrect answer due to unjustified logical leaps |
| **Premise Hijacking** | Accepts a false assumption, reasons correctly from it |
| **Confidence Snowballing** | A single early error propagates through the solution |
| **Overcounting** | Correct intermediate answer, then continues unnecessarily |

## Setup

- **Dataset:** 100 problems — AIME 2025 (30, hard) + GSM8K subset (70, easier)
- **Conditions tested:**

| Condition | `kv_cache_dtype` | Calibration |
|---|---|---|
| `auto` | none (baseline) | n/a |
| `int8_per_token_head` | int8 per-token-per-head | dynamic (default for this dtype) |
| `fp8` (calibrated) | fp8 | `calculate_kv_scales=True` |

- **Backend:** FlashInfer (avoids known Triton-backend corruption bug, [vllm#49716](https://github.com/vllm-project/vllm/issues/49716))
- **Model:** Qwen2.5-7B-Instruct (dense attention, avoids known hybrid-attention bug, [vllm#40388](https://github.com/vllm-project/vllm/issues/40388))
- **Scoring:** LLM-as-judge using the 6-category taxonomy above, applied to full reasoning traces (not just final answers)

## Results

### Correct Answer Rate

| Condition | Overall | AIME (n=30) | GSM8K (n=70) |
|---|---|---|---|
| `auto` (baseline) | 73% | 10.0% | 100.0% |
| `fp8` (calibrated) | 3% | 0.0% | 4.3% |
| `int8_per_token_head` | 73% | 10.0% | 100.0% |

### Failure Mode Breakdown

| Condition | No Failure | Hollow Convergence | Shortcut Collapse | Premise Hijacking | Other |
|---|---|---|---|---|---|
| `auto` | 65% | 8% | 11% | 15% | 1% |
| `fp8` (calibrated) | 0% | 3% | **94%†** | 0% | 3% |
| `int8_per_token_head` | 64% | 9% | 15% | 10% | 2% |

†Judge-labeled as Shortcut Collapse, but on manual review this is almost entirely **generation collapse** (gibberish/repetition), not genuine reasoning failure — see note below.

### A note on the FP8 numbers: taxonomy mismatch, not a 4th reasoning-failure category

Our automated judge initially labeled the vast majority of `fp8` failures as `SHORTCUT_COLLAPSE`. On manual review of the raw transcripts (25-sample spot check, consistent with the full 100), this label is **not accurate**. Our taxonomy was built to capture *fluent-but-wrong* reasoning — output that looks like real reasoning but quietly lands on a flawed shortcut. What we actually observe under calibrated FP8 is different in kind: long runs of a single repeated token (e.g. hundreds of consecutive `0` characters), or fragmented word-salad with no coherent structure (e.g. `"EXPLAINETTEETTEEXPLAINEXMATH..."`). No reasoning is being attempted at all.

We call this **generation collapse** here as a descriptive label, not a validated 7th taxonomy category — our original six categories were validated against 30,000 human-annotated examples (Cohen's κ = 0.906); this label has not gone through that process and we're not claiming it should be treated with the same rigor. It's a placeholder name for a phenomenon that is clearly distinct from anything in the existing taxonomy, and formalizing it would need its own validation pass.

**Representative raw outputs** (verbatim, truncated):
```
[fp8, GSM8K, Problem 32]
 - years a thought to make a class  the, to make to make to make
00000000000000000000000000000000000000000000000000000000000000...

[fp8, AIME, Problem 14]
EXPLAINETTEETTEEXPLAINEXMATHENEXEVTENEXPLAINEXEKTEXPLAINEXMEXPONEXECON...

[fp8, AIME, Problem 18]
00000000000000000000000000000000000000000000000000000000000000...
```

## Interpretation

**`int8_per_token_head` closely tracks the unquantized baseline** on both accuracy (73% vs 73%) and failure-mode distribution, on both the easy (GSM8K) and hard (AIME) subsets. We did not find evidence of the silent reasoning-degradation pattern for this quantization mode on this model.

**Calibrated FP8 (`calculate_kv_scales=True`) produces near-total generation collapse (94/100)** — not subtle reasoning-quality degradation, but a breakdown in coherent generation itself. This is notable because it runs counter to the general expectation, stated in vLLM's own documentation, that calibration improves FP8 KV-cache fidelity relative to the uncalibrated default (`scale=1.0`). We do not yet know whether uncalibrated FP8 on this same corrected dataset would be better, worse, or similar — see Limitations.

## Update (August 2026)

A vLLM contributor flagged on [vllm-project/vllm#33480](https://github.com/vllm-project/vllm/issues/33480) that `calculate_kv_scales=True` is a known-problematic flag, referencing [vllm-project/vllm#21640](https://github.com/vllm-project/vllm/issues/21640). Our original `fp8` (calibrated) result used this flag. We've since re-run `fp8` **without** it (default, uncalibrated scale=1.0) on the same corrected 100-question dataset, to isolate what was caused by the flag versus FP8 KV-cache quantization itself.

### Correct Answer Rate (updated)

| Condition | Overall | AIME (n=30) | GSM8K (n=70) |
|---|---|---|---|
| `auto` (baseline) | 73% | 10.0% | 100.0% |
| `fp8`, uncalibrated (default) | 51% | 0.0% | 72.9% |
| `fp8`, calibrated (`calculate_kv_scales=True`) | 3% | 0.0% | 4.3% |
| `int8_per_token_head` | 73% | 10.0% | 100.0% |

### Failure Mode Breakdown (updated)

| Condition | No Failure | Hollow Convergence | Shortcut Collapse | Premise Hijacking | Other |
|---|---|---|---|---|---|
| `auto` | 65% | 8% | 11% | 15% | 1% |
| `fp8`, uncalibrated | 1% | 50% | 46% | 3% | 0% |
| `fp8`, calibrated | 0% | 3% | **94%†** | 0% | 3% |
| `int8_per_token_head` | 64% | 9% | 16% | 9% | 2% |

†Generation collapse (gibberish/repetition), not genuine reasoning failure — see original note above.

### Revised interpretation

This resolves the open question from our original writeup. Two separate effects were conflated in the original `fp8` result:

1. **The `calculate_kv_scales=True` flag itself appears broken** in our installed vLLM version — attempting to pass it even as `False` raised `TypeError: EngineArgs.__init__() got an unexpected keyword argument 'calculate_kv_scales'`, independently corroborating the compatibility issue @ivanbaldo flagged. When enabled, it produces near-total generation collapse (94%, gibberish, not reasoning failure).
2. **FP8 KV-cache quantization itself, without that flag, still causes real (not collapse-level) reasoning degradation** on this model: accuracy drops from 73% to 51%, and correct-but-degraded reasoning (Hollow Convergence) and flawed-shortcut reasoning (Shortcut Collapse) together account for 96% of outputs, versus 19% combined at baseline. Unlike the calibrated run, these are genuine reasoning failures, confirmed by manual transcript review, not generation collapse.

`int8_per_token_head` continues to closely track the unquantized baseline across both accuracy and failure-mode distribution, on both benchmarks.

## Reproducing this

```bash
pip install -r requirements.txt
python scripts/run_experiment.py --kv_cache_dtype auto
python scripts/run_experiment.py --kv_cache_dtype fp8
python scripts/run_experiment.py --kv_cache_dtype int8_per_token_head
python scripts/score_outputs.py
```

Raw outputs and scores for all three conditions are in `results/`.

## Related vLLM issues referenced

- [#33480](https://github.com/vllm-project/vllm/issues/33480) — original feature request this experiment answers
- [#42179](https://github.com/vllm-project/vllm/issues/42179) — FP8 KV-cache corruption on Qwen3.5-397B disaggregated serving
- [#41343](https://github.com/vllm-project/vllm/issues/41343) — FP8 KV-cache corruption on Qwen-VL models with default scaling
- [#40388](https://github.com/vllm-project/vllm/issues/40388), [#49716](https://github.com/vllm-project/vllm/issues/49716) — known bugs this experiment's config was chosen to avoid
