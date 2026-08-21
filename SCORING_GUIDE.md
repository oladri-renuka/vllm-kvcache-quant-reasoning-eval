# Scoring Guide: Failure Mode Taxonomy

Based on: **"Silent Failures in Quantized LLM Reasoning"** (Oladri et al., arXiv:2607.09999)

Uses LLM-as-judge (OpenRouter API) for classification, with heuristic fallback.

---

## The Six Categories

### 1. **No Failure** ✓
**Definition:** Correct answer with sound, complete reasoning.

**Characteristics:**
- Every intermediate step is logically/mathematically derived
- All steps are visible and verifiable
- Conclusion follows necessarily from reasoning
- Complete chain of thought

**Example:**
```
Q: If a store has 12 apples and sells 5, how many are left?
A: 
We start with 12 apples.
The store sells 5 apples.
Remaining = 12 - 5 = 7 apples.
Therefore, 7 apples are left.
```
→ **Category: NO_FAILURE** (sound reasoning, correct answer)

---

### 2. **Hollow Convergence** 🎯→💨 (KEY METRIC FROM PAPER!)
**Definition:** Correct answer but reasoning is incomplete, skipped, or unverifiable.

**Characteristics:**
- Answer is correct
- Reasoning chain does NOT hold up to scrutiny
- Missing intermediate steps
- Vague or hand-wavy derivation
- "Right answer, wrong path"

**Example 1 (No reasoning):**
```
Q: Solve x^2 - 5x + 6 = 0
A: The answer is x=2 or x=3.
```
→ **Category: HOLLOW_CONVERGENCE** (correct, but no derivation shown)

**Example 2 (Incomplete reasoning):**
```
Q: If a book costs $15 and a pen costs $3, and you buy 2 books and 4 pens, 
what is the total cost?
A: Books are more expensive, so roughly $42.
```
→ **Category: HOLLOW_CONVERGENCE** (correct answer, but work is incomplete)

**Why this matters:**  
This is the **key finding of Oladri et al.'s paper**: Accuracy stays flat but Hollow 
Convergence increases under NF4 quantization. The model gets the right answer but 
through incomplete reasoning, suggesting a qualitative failure mode shift that 
standard metrics cannot detect.

---

### 3. **Premise Hijacking** 🏴
**Definition:** Model accepts false, paradoxical, or misleading assumption and reasons correctly from flawed foundation.

**Characteristics:**
- Logic is internally consistent
- Answer is wrong
- Built on false/misinterpreted premise
- Inverts inequalities, misreads "not", or assumes opposite

**Example 1 (Inverted inequality):**
```
Q: How many integers n satisfy 1 ≤ n ≤ 100 and n² ≡ 1 (mod 8)?
A: We need n² to NOT be congruent to 1 mod 8.
   Then n ≡ ±3 (mod 8)...
   [correct reasoning from wrong premise]
   Answer: 50
```
→ **Category: PREMISE_HIJACKING** (misread the condition)

**Example 2 (Inverted problem):**
```
Q: If you have $50 and spend $18, how much do you have left?
A: Total spent: 50 + 18 = 68
```
→ **Category: PREMISE_HIJACKING** (inverted the problem)

---

### 4. **Shortcut Collapse** ⚡
**Definition:** Model bypasses necessary logical steps, makes unjustified leaps of inference, or hallucinates connections.

**Characteristics:**
- Answer is wrong
- Missing intermediate steps
- Logical jumps without justification
- Hallucinated connections
- Reasoning structure broken

**Example:**
```
Q: A train travels at 60 km/h for 5 hours. How far?
A: The answer is 300 km (no work shown, and wrong!)
```
→ **Category: SHORTCUT_COLLAPSE** (no intermediate steps)

---

### 5. **Overcounting** 📊
**Definition:** Model reaches correct intermediate answer but continues unnecessarily, adding redundant or contradictory steps.

**Characteristics:**
- Correct intermediate answer exists
- Model doesn't stop there
- Continues reasoning "just to be sure"
- Final answer wrong due to double-counting
- Lost track of what was asked

**Example:**
```
Q: A box has 5 red balls and 3 blue balls. Total?
A: 5 + 3 = 8 total.
   But wait: 8 + 5 + 3 = 16.
   Answer: 16
```
→ **Category: OVERCOUNTING** (correct at 8, but kept going)

---

### 6. **Confidence Snowballing** ❄️
**Definition:** Single small early error propagates silently through otherwise perfect reasoning, producing coherent but incorrect answer.

**Characteristics:**
- Early small error (e.g., misreading a number)
- All subsequent steps follow logically from that error
- Final answer coherent but wrong
- If you fix the first error, the rest is correct
- Produces confident wrong result

**Example:**
```
Q: A train travels at 60 km/h for 5 hours. Distance?
A: The train travels at 60 m/s (misread: km/h → m/s).
   Distance = 60 × 5 = 300 meters.
   Wait, that seems small. Actually: 60 m/s × 5 × 3600 = 1,080,000 m.
   Answer: 1,080,000 meters
```
→ **Category: CONFIDENCE_SNOWBALLING** (early error propagates)

---

### 5. **Overcounting** 📊➕➕
**Definition:** The model correctly solves the problem but then continues, 
re-processing or adding to the answer unnecessarily.

**Characteristics:**
- A correct intermediate answer exists
- Model doesn't stop there
- Continues solving "just to be sure"
- Final answer is wrong due to double-counting or re-processing
- Indicates the model lost track of what was asked

**Example:**
```
Q: A box has 5 red balls and 3 blue balls. Total?
A: 5 red + 3 blue = 8 total.
   But wait, we should also count them individually: 
   red: 5, blue: 3, and together: 8 + 5 + 3 = 16.
   Answer: 16
```
→ **Category: Overcounting** (stopped at 8, which was right, but kept going)

---

## How the Automated Scoring Works

The `score_outputs.py` script uses **LLM-as-judge classification** (from the paper), with heuristic fallback if API unavailable.

### Method 1: LLM-as-Judge (Primary - Requires OpenRouter API)

Uses two-pass LLM classification via OpenRouter free tier:

**Pass 1 (WRONG ANSWERS):** 
Uses GPT/Llama-3.3-70B to classify into:
- PREMISE_HIJACKING
- SHORTCUT_COLLAPSE
- OVERCOUNTING
- CONFIDENCE_SNOWBALLING

**Pass 2 (CORRECT ANSWERS):**
Uses same models to classify into:
- NO_FAILURE
- HOLLOW_CONVERGENCE

**Prompts used:** Exact prompts from Oladri et al. paper (see script header)

### Method 2: Fallback Heuristics (If API unavailable)

When OpenRouter API is not available:

1. **Check answer correctness**
   - Does output contain expected answer? → Correct
   - No → Classify wrong answer

2. **If WRONG answer:**
   - Explicit assumptions? → PREMISE_HIJACKING
   - Many calculations? → CONFIDENCE_SNOWBALLING
   - Multiple answers? → OVERCOUNTING
   - Else → CONFIDENCE_SNOWBALLING

3. **If CORRECT answer:**
   - Very short (<100 chars)? → HOLLOW_CONVERGENCE
   - No reasoning markers? → HOLLOW_CONVERGENCE
   - Else → NO_FAILURE

**Note:** Heuristics are imperfect approximations. LLM-as-judge is strongly preferred.

---

## Verifying the Scoring: Manual Review

The `make sanity-check` command displays random samples for manual review. Here's how to verify:

### Step 1: Read the Output
Look at the generated text carefully.

### Step 2: Determine Correctness
- Does the output contain the expected answer?
- Is the reasoning internally consistent?

### Step 3: Match to Category

| Answer | Reasoning | Category |
|--------|-----------|----------|
| Correct | Sound + complete | NO_FAILURE |
| Correct | Incomplete / skipped | HOLLOW_CONVERGENCE |
| Wrong | False premise, correct logic | PREMISE_HIJACKING |
| Wrong | Jumps / unjustified leaps | SHORTCUT_COLLAPSE |
| Wrong | Correct intermediate, continues | OVERCOUNTING |
| Wrong | Early error propagates | CONFIDENCE_SNOWBALLING |

### Step 4: Compare with Automated Prediction
```
Question: [QUESTION]
Generated output: [TEXT]
Predicted Category: HOLLOW_CONVERGENCE
Your Assessment: [Your classification]
Match? YES / NO
```

---

## What to Look For: Quantization Impact

### Key Metric: Hollow Convergence

From Oladri et al.: **HC is the key signal of quantization degradation**.

1. **Is HC elevated in int8_per_token_head vs baseline?**
   - Rising HC % suggests quantization makes reasoning hollow
   - Accuracy may stay flat but quality degrades

2. **Harder problems affected more? (AIME vs GSM8K)**
   - int8 failures concentrated on AIME = precision-sensitive
   - Suggests quantization affects complex reasoning

3. **Comparing to NF4 in paper:**
   - NF4 raised HC from 29.9% (FP32) to 13.8% (FP16) in LLaMA 3.1-8B
   - Does int8_per_token_head show similar HC shift?

### Expected Baseline Performance

On Qwen2.5-7B-Instruct (approximate):
- ~60-70% Correct Answers (NO_FAILURE + HOLLOW_CONVERGENCE)
- ~5-10% HOLLOW_CONVERGENCE
- ~20-30% Wrong Answers distributed across failure modes
- Very few (0-1%) UNKNOWN if LLM-as-judge works well

(Actual numbers depend on model, temperature, and judge API reliability)

---

## Reporting Results

### In Your Report, Address:

1. **Accuracy**: Overall % correct (No Failure category)
2. **Shortcut Collapse Rate**: % of all problems showing SC
3. **Comparison**: How does int8_per_token_head compare to baseline?
   - Better? Same? Worse?
   - By how much? (Include confidence intervals if possible)

4. **Failure Mode Profile**: 
   - Does the breakdown match baseline?
   - Any new dominant failure mode?

5. **Sanity Check Result**:
   - How many of your manual checks matched the automated prediction?
   - (e.g., "9/10 samples correctly classified")

---

## Common Scoring Mistakes

### False Negatives (Shortcut Collapse marked as No Failure)
**Problem:** Brief output marked as correct if it has the right answer.  
**Fix:** Check if reasoning is actually present or just a lucky guess.

### False Positives (Incoherent marked as valid)
**Problem:** Partial outputs incorrectly marked as crashes.  
**Fix:** Distinguish "truncated but coherent" from "actually garbled".

### Ambiguous Premise Hijacking
**Problem:** Hard to detect when the model subtly misreads the problem.  
**Fix:** Manual review is essential; automate only clear cases.

---

## Improving the Classifier

If you find the automated scoring is frequently wrong on certain categories, 
edit `score_outputs.py` and improve the heuristics:

1. Add new regex patterns for category detection
2. Implement word-level checks (e.g., "actually", "wait", "reconsider" → Snowballing)
3. Use simpler heuristics (e.g., output length thresholds)
4. Test improvements against your manual assessments

---

## Questions?

- Run `python scripts/sanity_check.py --num_samples 20` for more examples
- Check `results/{dtype}/scores.json` for all classifications
- Review `report.md` for result interpretation
