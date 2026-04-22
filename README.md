# Black-Box Optimisation (BBO) Capstone Project

## Section 1: Project Overview

This repository documents my work on the Black‑Box Optimisation (BBO) capstone project. The challenge involves optimising several unknown functions under strict query constraints. Each function is treated as a black box: its analytical form, gradients, and noise characteristics are unknown, and information is revealed only through sequential query feedback.

The overall goal of the BBO capstone project is to efficiently **maximise the output of multiple unknown functions** while operating under limited evaluation budgets. This mirrors real‑world ML scenarios such as hyperparameter tuning, simulation optimisation and experimental design, where evaluations are expensive and uncertainty must be managed explicitly.

This project strengthens skills in Bayesian optimisation, uncertainty‑aware modelling, experiment tracking, automated decision pipelines, and advanced machine learning techniques including CNN-based surrogate modeling – all essential for designing production ML systems on cloud platforms.

Key Insight: Inspired by the Squirrel Switching Optimizer, this project extends Bayesian optimisation into a hybrid adaptive framework that dynamically selects between exploration, model-based search, and local refinement strategies.

---

## Section 2: Data Structure and Formats

### Initial Seed Data (Directory Structure)

`data/initial_data/function_1..function_8/initial_inputs.npy` and `initial_outputs.npy`

### Weekly Data (Matrix Format)

- `data/weekly/inputs.txt` - Historical query submissions
- `data/weekly/outputs.txt` - Corresponding function evaluation results

Each **row** corresponds to a week (week1..weekN) and contains values for **all 8 functions**.

### Supported Input Formats

**weekly_inputs.txt** - Portal tokens with 8 functions per row:

- Space or comma separated: `0.123456-0.654321, 0.111111-0.222222, ...`
- With week index: `1 0.123456-0.654321 0.111111-0.222222 ...`
- Python-style array rows: `[array([0.123456, 0.654321]), ...]`

**weekly_outputs.txt** - Function evaluation results:

- Float values: `1.23, -0.04, ...` (8 total)
- With week index: `1 1.23 -0.04 ...` (8 total)
- NumPy-style scalar rows: `[np.float64(1.23), np.float64(-0.04), ...]`

> The system automatically handles line wrapping and format variations for robust data loading.

---

## Section 3: Challenge Objectives

The objective is to **maximise each unknown function** while working with:

- Limited query budget (one per function per week)
- Sequential feedback only
- Unknown function structure and noise characteristics
- Increasing dimensionality (2D to 8D across functions)

This requires balancing learning the search space with exploiting promising regions through sophisticated acquisition functions and surrogate modeling.

---

## Section 4: Technical Approach and Strategy Evolution

### Strategy Development Across Weeks

**Week 1 – Structured Exploration:**  
Initial queries prioritised coverage and diversity. With no feedback available, points were chosen away from boundaries to reduce uncertainty, especially in higher‑dimensional functions.

**Week 2 – Adaptive Exploration vs Exploitation:**  
Week‑1 outputs allowed relative performance comparison. Most functions remained exploratory due to high uncertainty, while promising regions began targeted refinement.

**Week 3 – Model‑Driven Bayesian Optimisation:**  
Gaussian Process models fitted per function with automatic kernel selection. Expected Improvement acquisition function balances posterior mean μ(x) and uncertainty σ(x) for optimal query placement.

**Week 4 – Fully Automated Pipeline:**  
Complete Bayesian optimisation system with automatic kernel switching, per‑function exploration/exploitation tuning, diagnostic tracking, and automated history preservation.

**Week 5 – Progressive Analysis Integration:**  
Enhanced with comprehensive visualization system tracking week-by-week optimization progress, enabling detailed campaign analysis and strategy validation.

**Week 6 – Advanced Optimization and Refinement:**  
Implemented sophisticated parameter tuning and convergence analysis. Advanced acquisition function strategies with adaptive exploration parameters, multi-objective balancing across function portfolios, and enhanced uncertainty quantification for robust decision-making in final optimization phases.

**Week 7 – Hybrid Switching Optimization (Squirrel-Inspired):**
The optimisation framework was extended from a static Bayesian approach to a dynamic hybrid switching strategy, inspired by the Squirrel (Switching Optimizer). The system now adaptively switches between:

Exploration (random/global search)
Bayesian optimisation (model-based search)
Local refinement (greedy optimisation near best points)

An acquisition function portfolio (EI, PI, UCB) improves robustness, while late-stage optimisation focuses on local refinement to accelerate convergence.

**Week 8 – LLM-Aware Optimisation:**
With 17 data points available, the optimisation strategy is further extended to incorporate **LLM-centred considerations**, reflecting real-world behaviours such as tokenisation effects, prompt sensitivity, decoding variability, and attention limits.

At this stage, the optimisation landscape is treated as **partially non-smooth**, where small input changes can lead to disproportionate output variation. As a result, the framework shifts toward **controlled exploitation with targeted exploration**.

The system now introduces:

- **Controlled Exploitation Bias:**
  Increased focus on refining high-performing regions identified in earlier weeks, while maintaining selective exploration in uncertain areas.

- **Instability Detection and Response:**
  Functions exhibiting high variance across recent outputs are flagged as unstable, triggering additional exploration to avoid premature convergence.

- **Similarity-Based Penalty (Prompt Overfitting Proxy):**
  Repeated or near-duplicate query points are penalised to prevent over-exploitation of narrow regions, reflecting prompt overfitting behaviour observed in LLM systems.

- **Boundary Sensitivity Adjustment:**
  Query points near extreme boundaries are penalised to reduce instability risks associated with irregular response behaviour and tokenisation artefacts.

- **Variance-Aware Uncertainty Boosting:**
  The uncertainty term (σ) is dynamically increased for unstable regions, ensuring continued exploration where the response surface is unreliable.

**Week 9 – Scaling and Emergence-Aware Optimisation:**
With 18 data points now available, the optimisation strategy extends the Week 8 LLM-aware framework by incorporating **scaling pressure** and **emergent behaviour detection** into the query-selection logic.

At this stage, the main question is no longer only whether a region looks promising, but whether recent gains are reliable, transferable across dimensions, and robust to abrupt regime changes. The system therefore moves to a **scaling-aware hedge strategy** that balances local refinement with uncertainty-seeking behaviour when the response surface becomes unstable or unexpectedly shifts.

The system now introduces:

- **Emergence Scoring:**
  A z-score style emergence metric compares the latest observation against prior behaviour, helping identify sudden qualitative shifts in the optimisation landscape.

- **Ruggedness Detection:**
  Recent second-order output changes are monitored to estimate surface roughness, reducing the chance of overcommitting to brittle local spikes.

- **Dimension-Aware Scaling Pressure:**
  Higher-dimensional functions receive an adaptive uncertainty bonus so the optimiser does not become overconfident when sample density remains low relative to dimensionality.

- **Hedge Strategy Between BO and Refinement:**
  When instability, emergence, or scaling pressure rises, the pipeline switches from pure local refinement to a mixed candidate-generation mode combining local and global search.

- **Fast GP Fallback for High-Dimensional Late Stage:**
  For the largest functions in the later rounds, the Gaussian Process fitting path uses a lighter configuration to preserve robustness and runtime while still producing actionable uncertainty estimates.

- **Parser Hardening for Weekly Outputs:**
  The weekly loader now correctly reads `np.float64(...)` output rows without accidentally extracting spurious values, ensuring that the optimisation logic operates on the true historical feedback.

This enhancement aligns the optimisation process with real-world ML systems, where more data can reveal qualitatively new behaviour rather than simply reducing uncertainty.

**Week 10 – Recovery-Aware Trust-Region Optimisation:**  
With Week 9 results now available, the strategy is extended to explicitly detect **late-stage drawdowns** and recover from brittle over-exploitation. Instead of assuming the most recent region remains trustworthy, the optimiser now measures whether performance has fallen sharply from a recent peak and reacts by shifting toward a trust-region recovery mode.

The Week 10 update introduces:

- **Drawdown Detection:** Quantifies how far the latest observation has fallen from recent best performance.
- **Recovery Strategy:** If a function shows a strong late-stage drop, candidate generation centres on both the best historical point and the centroid of top-performing points rather than only the latest sample.
- **Trust-Region Scaling:** Local search radius is tightened or loosened according to dimensionality, strategy, and drawdown severity.
- **Consensus Bonus:** Candidates closer to the top-performing cluster receive a score bonus, improving stability when the landscape is noisy.
- **Last-Point Repulsion After Collapse:** When the latest query underperforms badly, the optimiser penalises points too close to that last location to avoid re-sampling a failing pocket.

This makes the Week 10 search more robust in functions where Week 9 revealed regressions after earlier gains.

### Methods and Architecture

- **Gaussian Process Regression**: RBF and Matérn kernels with automatic selection
- **Expected Improvement Acquisition**: Balances exploration and exploitation
- **Per-Function Tuning**: ξ (explore/exploit balance) and β (UCB weighting)
- **Progressive Visualization**: Week-by-week historical analysis system
- **Automated Data Management**: Robust loading with format auto-detection
- **CNN-Enhanced Optimization**: Deep learning surrogate models and landscape modeling
- **Advanced Parameter Tuning**: Sophisticated convergence analysis and adaptive exploration
- **Multi-Objective Portfolio Balancing**: Intelligent resource allocation across functions
- **Emergence Diagnostics**: Regime-shift detection and strategy switching metadata
- **Scaling-Aware Candidate Search**: Dimension-sensitive uncertainty and hedge-based query generation

---

## Section 5: System Architecture

For detailed technical architecture, component descriptions, and system diagrams, see [docs/architecture.md](docs/architecture.md).

---

## Section 6: Running the System

### Installation and Setup

```bash
pip install -r requirements.txt
python scripts/run_week.py --initial_dir data/initial_data --weekly_dir data/weekly
```

**Optional CNN Enhancement:**

```bash
# For CNN-based surrogate modeling and landscape optimization
pip install torch torchvision matplotlib scipy
```

### Visualization Generation

**Complete Campaign Analysis:**

```bash
python scripts/visualize_progress.py
```

**Progressive Week-by-Week History:**

```bash
python scripts/progressive_visualize.py
```

### Generated Outputs

- **artifacts/visualizations/**: Complete optimization campaign analysis
- **artifacts/progressive_visualizations/**: Historical progression showing how optimization looked at each week
  - `week1/`: Cumulative view through Week 1
  - `week2/`: Cumulative view through Week 2
  - `week5/`: Cumulative view through Week 5
  - `week6/`: Advanced optimization and refinement analysis
  - `week7/`: Hybrid switching optimization analysis
  - `week8/`: LLM-Aware Optimisation
  - `week9/`: Scaling and Emergence-Aware Optimisation
  - `week10/`: Recovery-Aware Trust-Region Optimisation
- **artifacts/submissions/**: Portal-ready weekly query files

Each week folder contains:

- Function progress trajectories
- Week-to-week improvement analysis
- Performance summaries and success rates
- Optimization landscape heatmaps

---

## Section 7: Key Features and Enhancements

### Robust Data Management

- Automatic line-wrapping detection and correction
- Multiple input format support (portal tokens, arrays, NumPy scalar rows)
- Week numbering validation and consistency checks
- Safe parsing for `np.float64(...)` output records

### Advanced Optimization Features

- **Week 9 Scaling and Emergence-Aware Enhancements**: Emergence scoring, ruggedness penalties, dimension-aware scaling pressure, and hedge-based strategy switching
- **Week 8 LLM-Aware Optimisation Enhancements**: Instability-aware acquisition, similarity penalties, and boundary control mechanisms reflecting tokenisation effects and prompt sensitivity
- **Week 6 Sophisticated Parameter Tuning**: Adaptive exploration with convergence analysis
- **Multi-Objective Portfolio Balancing**: Intelligent resource allocation across function portfolios
- **Enhanced Uncertainty Quantification**: Robust decision-making with uncertainty boosting
- **CNN Integration Framework**: Deep learning surrogate models and landscape optimization
- **Hybrid Ensemble Methods**: CNN-GP combinations for enhanced modeling capability

### Week 7 Switching Optimisation Enhancements

- Squirrel-inspired strategy switching
- Stagnation detection mechanism
- Hybrid candidate generation (local + global)
- Acquisition portfolio (EI, PI, UCB)
- Improved convergence with 16+ data points

### Progressive Analysis System

- Week-by-week historical visualization preservation
- Cumulative progress tracking with consistent scaling
- Change detection and improvement analysis
- Advanced convergence monitoring and stability scoring
- Emergence and scaling diagnostics captured in pipeline reports

### Modern ML Integration

- **CNN Surrogate Models**: Deep learning-based function approximation
- **CNN Landscape Modeling**: Spatial optimization for 2D functions
- **Monte Carlo Uncertainty**: Enhanced uncertainty quantification
- **Transfer Learning**: Cross-campaign knowledge transfer

### Modular Architecture

- Clean separation of concerns (data, models, visualization)
- Extensible pipeline for new acquisition functions and CNN methods
- Comprehensive error handling and validation
- Seamless integration between classical and deep learning approaches

---

This README reflects the complete BBO optimization system with enhanced visualization capabilities, robust data management, Week 6 advanced optimization features, Week 7 hybrid switching optimization, Week 8 LLM-aware optimisation enhancements, and Week 9 scaling and emergence-aware optimisation updates.

The system now captures both classical black-box optimisation behaviour and the non-smooth, uncertain characteristics of LLM-driven systems, incorporating tokenisation effects, prompt sensitivity, attention limitations, scaling pressure, and emergent regime shifts into the optimisation framework.

## Documentation

- [Datasheet](docs/datasheet.md)
- [Model Card](docs/model_card.md)

## Additional Resources

- **[Technical Architecture](docs/architecture.md)**: Comprehensive system architecture with Mermaid diagrams
- **[CNN Integration Guide](docs/cnn_integration_guide.md)**: Deep learning enhancement documentation
- **[Week 6 Advanced Demo](scripts/demo_week6_advanced.py)**: Sophisticated optimization features demonstration
