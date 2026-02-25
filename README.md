# Black-Box Optimisation (BBO) Capstone Project

## Section 1: Project Overview

This repository documents my work on the Black‑Box Optimisation (BBO) capstone project. The challenge involves optimising several unknown functions under strict query constraints. Each function is treated as a black box: its analytical form, gradients, and noise characteristics are unknown, and information is revealed only through sequential query feedback.

The overall goal of the BBO capstone project is to efficiently **maximise the output of multiple unknown functions** while operating under limited evaluation budgets. This mirrors real‑world ML scenarios such as hyperparameter tuning, simulation optimisation and experimental design, where evaluations are expensive and uncertainty must be managed explicitly.

This project supports my career as a lead developer / solution architect by strengthening skills in Bayesian optimisation, uncertainty‑aware modelling, experiment tracking and automated decision pipelines – all useful when designing production ML systems on cloud platforms.

---

## Section 2: Inputs and Outputs

Each iteration consists of submitting one query per function and receiving a scalar response.

### Inputs

- Query format: `x1-x2-x3-...-xn`
- Each value in `[0,1]`
- Six decimal precision
- Dimensions: 2D → 8D
- Constraint: One query per function per week

Example:
`0.372451-0.684219`

### Outputs

- Single real‑valued score
- Unknown scale and noise
- Used to update surrogate models and guide next queries

---

## Section 3: Challenge Objectives

The objective is to **maximise each unknown function** while working with:

- Limited query budget
- Sequential feedback only
- Unknown function structure
- Increasing dimensionality

This requires balancing learning the search space with exploiting promising regions.

---

## Section 4: Technical Approach

### Strategy Evolution

**Week 1 – Structured Exploration:**  
Initial queries prioritised coverage and diversity. With no feedback available, points were chosen away from boundaries to reduce uncertainty, especially in higher‑dimensional functions.

**Week 2 – Adaptive Exploration vs Exploitation:**  
Week‑1 outputs allowed relative performance comparison. Most functions remained exploratory due to high uncertainty, while a few promising regions were refined locally.

**Week 3 – Model‑Driven Bayesian Optimisation:**  
A Gaussian Process model was fitted per function. Kernel choice and hyperparameters were tuned automatically using log‑marginal likelihood. Queries were selected using Expected Improvement based on posterior mean μ(x) and uncertainty σ(x).

**Week 4 – Fully Automated Optimisation Pipeline:**  
The workflow evolved into a complete Bayesian optimisation system with automatic kernel switching, per‑function exploration/exploitation tuning using ξ and β, diagnostic tracking, and automated history storage. High‑performing functions moved to exploitation, while uncertain high‑dimensional ones continued exploration.

**Week 5 – Architecture‑Inspired Strategy:**  
With more data available, ideas from neural network design influenced optimisation. Functions were treated like layered models: feature importance from GP length‑scales identified relevant dimensions, acquisition tuning acted like learning‑rate scheduling, and model comparison mirrored architecture selection. Exploration focused on uncertain “feature hierarchies,” while exploitation refined high‑performing regions.

### Methods Used

- Gaussian Process regression with RBF and Matérn kernels
- Automatic kernel selection via marginal likelihood
- Expected Improvement acquisition
- Per‑function tuning using ξ (explore/exploit) and β (UCB‑style weighting)
- Diagnostics tracking kernel, ξ, β and exploration ratio
- Automated weekly history storage and plotting

### Exploration vs Exploitation

Exploration and exploitation are controlled explicitly through acquisition parameters. Small ξ values focus on improving known good regions, while larger ξ encourages sampling uncertain areas. High‑dimensional functions remain exploratory longer because sparse data increases uncertainty.

### Possible Extensions

Support Vector Machines could classify high vs low performance regions using soft‑margin SVMs. Kernel SVMs would help detect non‑linear boundaries. Neural networks could also be explored as surrogate models when data volume grows.

---

## Repository Architecture

```
ml-ai-capstone-project/
│
├── bbo_gp_weekly_generator.py
├── plots/
│   ├── func01/
│   ├── func02/
│   └── ...
├── history/
└── README.md
```

### Architecture Diagram

```
Historical Data + Weekly Results
            │
            ▼
     Gaussian Process Models
            │
Kernel Selection + Hyperparameter Tuning
            │
            ▼
   Acquisition Function (EI / UCB)
            │
Explore vs Exploit Decision (ξ, β)
            │
            ▼
       Next Week Queries
            │
            ▼
     Portal Submission → New Data
```

---

This README reflects my evolving optimisation strategy and will continue to be updated as the BBO capstone progresses.
