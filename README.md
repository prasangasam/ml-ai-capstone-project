# Black-Box Optimisation (BBO) Capstone Project

## Section 1: Project Overview

This repository documents my work on the Black‑Box Optimisation (BBO) capstone project. The challenge involves optimising several unknown functions under strict query constraints. Each function is treated as a black box: its analytical form, gradients, and noise characteristics are unknown, and information is revealed only through sequential query feedback.

The overall goal of the BBO capstone project is to efficiently **maximise the output of multiple unknown functions** while operating under limited evaluation budgets. This mirrors real‑world ML scenarios such as hyperparameter tuning, simulation optimisation and experimental design, where evaluations are expensive and uncertainty must be managed explicitly.

This project strengthening skills in Bayesian optimisation, uncertainty‑aware modelling, experiment tracking and automated decision pipelines – all useful when designing production ML systems on cloud platforms.

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

**weekly_outputs.txt** - Function evaluation results:

- Float values: `1.23, -0.04, ...` (8 total)
- With week index: `1 1.23 -0.04 ...` (8 total)

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

### Methods and Architecture

- **Gaussian Process Regression**: RBF and Matérn kernels with automatic selection
- **Expected Improvement Acquisition**: Balances exploration and exploitation
- **Per-Function Tuning**: ξ (explore/exploit balance) and β (UCB weighting)
- **Progressive Visualization**: Week-by-week historical analysis system
- **Automated Data Management**: Robust loading with format auto-detection

---

## Section 5: Repository Architecture

```
bbo_capstone_matrix_weekly_project/
│
├── scripts/
│   ├── run_week.py                    # Main BBO execution pipeline
│   ├── visualize_progress.py          # Complete campaign visualization
│   └── progressive_visualize.py       # Week-by-week historical graphs
│
├── src/
│   └── bbo/                           # Core optimization modules
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── data_loader.py             # Robust data loading with auto-format
│       ├── gp.py                      # Gaussian Process implementation
│       ├── io.py                      # Input/output utilities
│       ├── pipeline.py                # Main optimization pipeline
│       └── strategy.py                # Acquisition strategies
│
├── data/
│   ├── initial_data/                  # Seed data (NPY format)
│   │   └── function_1..8/
│   └── weekly/                        # Historical campaign data
│       ├── inputs.txt                 # Query history (portal format)
│       └── outputs.txt                # Evaluation results
│
├── artifacts/
│   ├── visualizations/                # Complete campaign analysis
│   └── progressive_visualizations/    # Week-by-week historical graphs
│       ├── week1/                     # Week 1 cumulative view
│       ├── week2/                     # Week 2 cumulative view
│       ├── ...                        # Progressive weekly analysis
│       └── README.md                  # Visualization guide
│
├── docs/
│   └── architecture.md                # Technical documentation
│
├── requirements.txt
└── README.md
```

### Architecture Flow

```
Historical Data (inputs.txt/outputs.txt)
              │
              ▼
    Enhanced Data Loader (auto-format detection)
              │
              ▼
     Gaussian Process Models (per function)
              │
Kernel Selection + Hyperparameter Tuning
              │
              ▼
   Acquisition Function (Expected Improvement)
              │
  Explore vs Exploit Decision (ξ, β tuning)
              │
              ▼
        Next Week Queries
              │
              ▼
     Portal Submission → New Data
              │
              ▼
   Progressive Visualization System
```

---

## Section 6: Running the System

### Installation and Setup

```bash
pip install -r requirements.txt
python scripts/run_week.py --initial_dir data/initial_data --weekly_dir data/weekly
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
  - `week5/`: Complete campaign view

Each week folder contains:

- Function progress trajectories
- Week-to-week improvement analysis
- Performance summaries and success rates
- Optimization landscape heatmaps

---

## Section 7: Key Features and Enhancements

### Robust Data Management

- Automatic line-wrapping detection and correction
- Multiple input format support (portal tokens, arrays)
- Week numbering validation and consistency checks

### Progressive Analysis System

- Week-by-week historical visualization preservation
- Cumulative progress tracking with consistent scaling
- Change detection and improvement analysis

### Modular Architecture

- Clean separation of concerns (data, models, visualization)
- Extensible pipeline for new acquisition functions
- Comprehensive error handling and validation

---

This README reflects the complete BBO optimization system with enhanced visualization capabilities and robust data management, supporting thorough analysis of the optimization campaign's effectiveness and progression.
