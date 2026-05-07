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

- 12 rounds of queries
- 8 functions (2D to 8D)
- 1 query per function per round

### Size

- 96 query points
- 96 function evaluations

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
- **Round 11:** Cluster-aware refinement
- **Round 12:** PCA-inspired variance-aware refinement

### Week 10 Strategy

- Drawdown detection (performance drop)
- Recovery sampling using:
  - Best historical point
  - Centroid of top-performing points
- Trust-region refinement
- Avoidance of recent poor regions

### Week 11 Strategy

- Cluster detection around nearby high-performing query points
- Centroid-guided local refinement
- Filtering of isolated outliers and noisy regions
- Boundary tightening around promising local clusters
- Reduced attention to repeatedly weak or inconsistent regions

### Week 12 Strategy (PCA-Informed Refinement)

- Uses 21 accumulated data points per function to identify dominant sources of variation
- Treats high-impact directions as PCA-like principal components of the optimisation landscape
- Reduces movement in low-impact or redundant dimensions to simplify the search
- Applies controlled perturbations around high-performing clusters rather than broad random sampling
- Balances exploitation of promising regions with limited exploration to avoid local optima
- Interprets optimisation outcomes through variance, redundancy, and dimensionality-reduction concepts

This stage marks a shift from heuristic search to structure-driven optimisation: the model now uses learned variation patterns to decide where each new query is most informative.

### Week 13 Strategy (RL Feedback-Adaptive Refinement)

- Uses 22 accumulated data points per function after the latest weekly feedback
- Treats each submitted query as an action and each output change as a reward signal
- Applies epsilon-style exploration: more exploration after weak, unstable, or negative feedback, and more exploitation after reliable improvement
- Uses a credit-assignment proxy to revisit regions linked to the strongest recent improvement
- Keeps global probes to avoid premature convergence while refining around historically rewarded regions
- Records RL diagnostics such as recent reward, reward rate, and effective epsilon

This stage connects the optimisation process to reinforcement learning: the strategy is updated by feedback, not only by static surrogate-model predictions.

### Time Frame

- Data collected over 13 sequential rounds
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
