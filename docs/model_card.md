# Model Card: Adaptive Cluster-Aware Optimisation Strategy

## 1. Overview (NAME + TYPE + VERSION)

- Model Name: Adaptive Cluster-Aware Optimiser
- Type: Sequential black-box optimisation strategy
- Version: 1.1

---

## 2. Intended Use (SUITABLE + AVOID CASES)

### Suitable Tasks

- Black-box optimisation problems
- Low evaluation budget scenarios
- Sequential decision-making tasks

### Cases to Avoid

- High-noise environments
- Problems requiring global optimality guarantees
- Large-scale automated optimisation

---

## 3. Strategy Details (TECHNIQUES + EVOLUTION)

### Strategy Across 11 Rounds

**Rounds 1–3 (Exploration):**

- Random and diverse sampling
- Goal: maximise search space coverage

**Rounds 4–7 (Exploitation):**

- Focus on high-performing regions
- Reduced exploration

**Rounds 8–10 (Refinement & Recovery):**

- Introduced recovery mechanism
- Balanced refinement and correction

**Round 11 (Cluster-Aware Refinement):**

- Identified recurring high-performing regions
- Used centroid trends to guide local refinement
- Treated isolated strong or weak points as possible noise
- Tightened boundaries around promising clusters

### Techniques Used

- Local refinement (trust-region search)
- Exploration of sparse regions
- Recovery from performance drop
- Avoidance of recent poor points
- Use of Gaussian Process surrogate model
- High-performer cluster detection
- Centroid-guided query selection
- Noise and outlier filtering

### Decision Logic

At each round:

1. Analyse historical performance
2. Identify trend:
   - Improvement → refine
   - Uncertainty → explore
   - Decline → recover
3. Select next query accordingly

---

## 4. Performance (RESULTS + METRICS)

### Metrics Used

- Function output values (maximisation)
- Improvement trends across rounds
- Stability of results

### Summary of Results

- Strong improvement in early rounds
- Some instability due to local optima
- Recovery strategy improved late-stage performance
- Cluster-aware refinement improved consistency by prioritising recurring promising regions
- Performance varied by function dimensionality

---

## 5. Assumptions and Limitations (ASSUMPTIONS + FAILURE MODES)

### Key Assumptions

- Function landscapes are relatively smooth
- Historical high-performing regions remain useful
- Local optimisation leads to improvement
- Nearby high-performing points are more reliable than isolated one-off results

### Limitations

- Limited number of queries (80 total)
- Risk of local optima convergence
- Sparse coverage in high dimensions
- Dependence on early sampling decisions

### Potential Failure Modes

- Getting stuck in local optima
- Over-exploitation of early good regions
- Missing better regions due to limited exploration
- Mistaking a noisy group of points for a meaningful cluster

---

## 6. Ethical Considerations (TRANSPARENCY + REPRODUCIBILITY)

Transparency in this approach supports:

- Reproducibility of optimisation decisions
- Clear understanding of how queries are selected
- Identification of sampling bias
- Clear explanation of how clusters, outliers, and noisy regions influence query selection

This improves:

- Trust in the optimisation process
- Adaptability to real-world optimisation problems

---

## 7. Reflection on Clarity

The model card clearly explains:

- How decisions are made
- How the strategy evolves
- What assumptions are used

Additional technical details (e.g. hyperparameters) could improve reproducibility further, but the current structure is sufficient for understanding and evaluating the approach.
