# Model Card: Adaptive RL Feedback-Informed Optimisation Strategy

## 1. Overview (NAME + TYPE + VERSION)

- Model Name: Adaptive RL Feedback-Informed Optimiser
- Type: Sequential black-box optimisation strategy
- Version: 1.3

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

### Strategy Across 13 Rounds

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

**Round 12 (PCA-Inspired Variance-Aware Refinement):**

- Analysed accumulated data to identify dominant sources of variation
- Prioritised dimensions that appear to drive the largest performance changes
- Reduced variation in low-impact dimensions as an implicit dimensionality reduction step
- Applied controlled perturbations along promising directions
- Strengthened exploitation while preserving limited exploration

**Round 13 (RL Feedback-Adaptive Refinement):**

- Treated query choices as actions and week-to-week output changes as rewards
- Adjusted exploration using an epsilon-style control parameter
- Used credit assignment to revisit regions associated with stronger improvement
- Increased recovery/exploration when recent rewards were weak or unstable
- Preserved human-readable diagnostics for strategy reflection

### Techniques Used

- Local refinement (trust-region search)
- Exploration of sparse regions
- Recovery from performance drop
- Avoidance of recent poor points
- Use of Gaussian Process surrogate model
- High-performer cluster detection
- Centroid-guided query selection
- Noise and outlier filtering
- PCA-inspired variance analysis
- Implicit dimensionality reduction through feature-importance patterns
- Directional search along dominant performance gradients
- RL-style reward tracking and feedback-driven policy updates
- Epsilon-style exploration/exploitation balancing
- Credit-assignment proxy for recently successful query regions

### Decision Logic

At each round:

1. Analyse historical performance
2. Identify trend:
   - Improvement → exploit rewarded regions and refine along dominant directions
   - Uncertainty → increase epsilon and explore selectively
   - Decline → recover using structured sampling and avoid failed pockets
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
- PCA-inspired refinement improved query efficiency by focusing on high-variance directions and reducing redundant movement
- RL feedback-adaptive refinement made the final strategy more responsive to recent rewards, instability, and exploration-exploitation balance
- Performance varied by function dimensionality

---

## 5. Assumptions and Limitations (ASSUMPTIONS + FAILURE MODES)

### Key Assumptions

- Function landscapes are relatively smooth
- Historical high-performing regions remain useful
- Local optimisation leads to improvement
- Nearby high-performing points are more reliable than isolated one-off results
- High-variance input directions contain more useful optimisation information than low-variance redundant directions

### Limitations

- Limited number of queries (104 total after Round 13)
- Risk of local optima convergence
- Sparse coverage in high dimensions
- Dependence on early sampling decisions

### Potential Failure Modes

- Getting stuck in local optima
- Over-exploitation of early good regions
- Missing better regions due to limited exploration
- Mistaking a noisy group of points for a meaningful cluster
- Over-simplifying the search by reducing variation in dimensions that may still contain hidden interactions
- Overreacting to recent reward signals if the latest feedback is noisy

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

## 7. Reflection: Exploration, Exploitation and RL Feedback

With the latest weekly data added, the strategy now behaves more like an RL feedback loop. Each submitted query is treated as an action, and each observed change in output is treated as a reward signal that updates the next decision.

This creates a practical exploration-exploitation trade-off:

- Exploitation improves by refining around regions that have repeatedly produced strong or improving rewards
- Exploration increases when recent rewards are weak, unstable, or declining
- Credit assignment helps connect successful improvements to earlier query locations
- Feedback-driven updates make the approach more adaptive than a fixed search heuristic
- The remaining risk is overreacting to noisy short-term feedback, so global probes and human review remain important

Future work should continue adapting the trust region and re-checking low-variance dimensions so that simplification does not remove useful signal.

---

## 8. Reflection on Clarity

The model card clearly explains:

- How decisions are made
- How the strategy evolves
- What assumptions are used

Additional technical details (e.g. hyperparameters) could improve reproducibility further, but the current structure is sufficient for understanding and evaluating the approach.
