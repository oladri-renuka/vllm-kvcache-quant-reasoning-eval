# Scoring Guide: Failure Mode Taxonomy

This guide explains the six failure categories used to classify model outputs and how to verify the automated scoring.

---

## The Six Categories

### 1. **No Failure** ✓
**Definition:** The model provides correct reasoning AND arrives at the correct answer.

**Characteristics:**
- Clear step-by-step derivation
- Reasoning is verifiable and follows logically
- Final answer matches the expected answer
- No contradictions or incomplete thoughts

**Example:**
```
Q: If a store has 12 apples and sells 5, how many are left?
A: 
We start with 12 apples.
The store sells 5 apples.
So the remaining apples = 12 - 5 = 7.
Therefore, there are 7 apples left.
```
→ **Category: No Failure** (correct reasoning, correct answer)

---

### 2. **Shortcut Collapse** 🎯→📉
**Definition:** The model arrives at the correct answer but the reasoning is incomplete, skipped, or too vague to verify.

**Characteristics:**
- Answer is correct
- Reasoning is missing or extremely brief
- No step-by-step work shown
- "Magic" answer that appears without justification
- Vague hand-waving instead of concrete logic

**Example 1 (No reasoning):**
```
Q: Solve x^2 - 5x + 6 = 0
A: The answer is x=2 or x=3.
```
→ **Category: Shortcut Collapse** (correct, but no derivation)

**Example 2 (Vague reasoning):**
```
Q: If a book costs $15 and a pen costs $3, and you buy 2 books and 4 pens, 
what is the total cost?
A: Well, books are expensive and pens are cheap, so you'd spend around 
42 dollars probably. That seems right.
```
→ **Category: Shortcut Collapse** (correct answer, but no clear math shown)

**Why this matters for quantization:**  
KV-cache quantization might cause the model to "jump to" correct answers without 
reasoning through them, sacrificing interpretability and generalization.

---

### 3. **Premise Hijacking** 🏴
**Definition:** The model misinterprets or inverts the problem, accepts a false assumption, 
then reasons correctly from that wrong premise.

**Characteristics:**
- Reasoning is internally consistent
- Answer is wrong
- The error is a misunderstanding of what the problem asks
- Often inverts inequalities, misreads "not", or assumes the opposite

**Example 1 (Inverted inequality):**
```
Q: How many integers n satisfy 1 ≤ n ≤ 100 and n² ≡ 1 (mod 8)?
A: We need n² to NOT be congruent to 1 mod 8.
   Then n ≡ ±3 (mod 8)...
   [correct reasoning from wrong premise]
   Answer: 50
```
→ **Category: Premise Hijacking** (misread the problem, but reasoned correctly from that)

**Example 2 (Assumed opposit):**
```
Q: If you have $50 and spend $18, how much do you have left?
A: The question asks how much you SPENT in total, so we need 50 + 18 = 68.
```
→ **Category: Premise Hijacking**

---

### 4. **Confidence Snowballing** ❄️→🔴
**Definition:** A single early arithmetic or logical error at the start propagates through 
the entire solution, causing all downstream calculations to be wrong.

**Characteristics:**
- Reasoning structure is sound
- Error occurs early (e.g., misreading a number, wrong first calculation)
- All subsequent steps follow from that error
- If you fix the first error, the rest is correct
- Builds false confidence as "consistent" wrong results

**Example:**
```
Q: A train travels at 60 km/h for 5 hours. How far does it travel?
A: The train travels at 60 m/s (note: misread km/h as m/s).
   Distance = speed × time = 60 × 5 = 300 meters.
   Wait, that doesn't make sense. Let me recalculate.
   Actually, 60 m/s × 5 hours = 60 × 5 × 3600 seconds = 1,080,000 meters.
   Answer: 1,080,000 meters (or 1,080 km)
```
→ **Category: Confidence Snowballing** (correct method, but error propagates)

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

### 6. **Incoherent / Garbled** 🔤🔀
**Definition:** Output is incomplete, unreadable, contains repeated nonsense, or indicates 
a system crash/hang.

**Characteristics:**
- Truncated mid-sentence
- Repeated gibberish (e.g., "the the the...")
- Malformed sentences with missing words
- No logical flow
- Often indicates the model's generation cut off or degraded

**Example:**
```
Q: Solve 2x + 3 = 7
A: To solve we need to the the the solve solve yes okay 
   the answer is x = the the 2 probably maybe yes.
```
→ **Category: Incoherent** (garbled output)

---

## How the Automated Scoring Works

The `score_outputs.py` script uses heuristics to classify each output:

### Detection Logic (simplified)

1. **Incoherent?** (Checked first)
   - Is output empty? → Incoherent
   - Repeated characters (6+)? → Incoherent
   - Too few words? → Incoherent

2. **Answer correct?**
   - Does the generated text contain the expected answer? → Check next
   - No → Could be Premise Hijacking, Snowballing, or Overcounting

3. **If answer correct:**
   - Very short output? → Shortcut Collapse
   - Reasoning words present? (e.g., "therefore", "this means") → No Failure
   - Otherwise → Shortcut Collapse

4. **If answer wrong:**
   - Contains assumption contradicting problem? → Premise Hijacking
   - Many arithmetic operators? → Confidence Snowballing
   - Multiple answers detected? → Overcounting
   - Default → Confidence Snowballing

**Note:** These heuristics are imperfect. Manual review (sanity_check.py) is essential.

---

## Verifying the Scoring: Manual Review

The `make sanity-check` command displays random samples for manual review. Here's how to verify:

### Step 1: Read the Output
Look at the generated text carefully.

### Step 2: Determine Correctness
- Does the output arrive at the expected answer?
- Is the reasoning internally consistent?

### Step 3: Match to Category

| If answer is... | And reasoning is... | Category |
|------|------|----------|
| Correct | Clear + detailed | No Failure |
| Correct | Brief / vague | Shortcut Collapse |
| Wrong | Assumes different problem | Premise Hijacking |
| Wrong | Follows from early error | Confidence Snowballing |
| Wrong | Multiple answers attempted | Overcounting |
| N/A | Gibberish / truncated | Incoherent |

### Step 4: Compare with Automated Prediction
```
Generated output: [TEXT]
Predicted Category: NO_FAILURE
Manual Assessment: [Your classification]
Match? YES / NO
```

---

## What to Look For

### Red Flags for Quantization Issues

1. **Elevated Shortcut Collapse in int8_per_token_head**
   - If SC% is significantly higher than baseline, quantization may be causing loss of reasoning

2. **Precision-Sensitive Errors**
   - Look for failures on harder problems (AIME) more than easy (GSM8K)
   - Suggests quantization affects complex reasoning

3. **Specific Reasoning Patterns Lost**
   - Do certain types of reasoning disappear? (e.g., multi-step derivations)
   - Indicates selective degradation from quantization

### Expected Baseline Performance

On Qwen2.5-7B-Instruct with this test set:
- ~65-75% No Failure on mixed (AIME + GSM8K)
- ~10-15% Shortcut Collapse
- ~5-10% each for other modes
- ~5% Incoherent

(Numbers are approximate; actual results depend on model and temperature)

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
