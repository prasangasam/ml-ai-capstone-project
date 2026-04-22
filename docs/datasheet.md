# Datasheet: BBO Capstone Query Dataset

## Dataset Name

BBO Capstone Query History Dataset (v1.0)

---

## 1. Motivation (WHY + TASK SUPPORT)

This dataset was created to support a **black-box optimisation (BBO)** task involving eight unknown functions.

The objective of the task is to:

- Iteratively select query points
- Maximise function outputs
- Operate under a limited evaluation budget

The dataset enables:

- Analysis of optimisation strategies across rounds
- Evaluation of exploration vs exploitation trade-offs
- Reproducibility of sequential decision-making

---

## 2. Composition (CONTENTS + SIZE + FORMAT + GAPS)

### Contents

- 10 rounds of queries
- 8 functions (2D to 8D)
- 1 query per function per round

### Size

- 80 query points
- 80 function evaluations

### Format

Each entry contains:

- `function_id`
- `round`
- `input` (vector of floats in [0,1])
- `output` (function value)

### Gaps and Limitations

- Sparse sampling in higher-dimensional functions (6D–8D)
- Uneven distribution (clustered near high-performing regions)
- Limited exploration in later rounds due to optimisation focus

---

## 3. Collection Process (HOW + STRATEGY + TIMEFRAME)

### How Queries Were Generated

Queries were generated sequentially using the optimisation pipeline:

- `src/bbo/pipeline.py`
- `src/bbo/strategy.py`

### Strategy Used

- **Rounds 1–3:** Broad exploration
- **Rounds 4–7:** Mixed exploration and exploitation
- **Rounds 8–10:** Refinement and recovery

### Week 10 Strategy

- Drawdown detection (performance drop)
- Recovery sampling using:
  - Best historical point
  - Centroid of top-performing points
- Trust-region refinement
- Avoidance of recent poor regions

### Time Frame

- Data collected over 10 sequential rounds
- Each round depends on results from previous rounds

---

## 4. Preprocessing and Uses (TRANSFORMATIONS + USE CASES)

### Transformations

- No major preprocessing applied
- Data kept in raw numerical format
- Values rounded to 6 decimal places for submission

### Intended Uses

- Evaluating optimisation strategies
- Reproducing query decisions
- Analysing sequential search behaviour

### Inappropriate Uses

- Not suitable for supervised learning
- Not representative of real-world datasets
- Not suitable for generalising function behaviour

---

## 5. Distribution and Maintenance (WHERE + TERMS + OWNER)

### Availability

- Stored in this repository:
  `data/query_history.json`

### Terms of Use

- Publicly available for educational use
- No restrictions beyond academic use

### Maintenance

- Maintained by the repository owner
- No further updates planned after submission
