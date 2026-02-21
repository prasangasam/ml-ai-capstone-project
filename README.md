[readme.md](https://github.com/user-attachments/files/25130306/readme.md)

# Black-Box Optimisation (BBO) Capstone Project

## Section 1: Project Overview

This repository documents my work on the Black-Box Optimisation (BBO) capstone project. The challenge involves optimising several unknown functions under strict query constraints. Each function is treated as a black box: its analytical form, gradients, and noise characteristics are unknown, and information is revealed only through sequential query feedback.

The overall goal of the BBO capstone project is to efficiently **maximise the output of multiple unknown functions** while operating under limited evaluation budgets. This closely reflects real-world machine learning scenarios such as hyperparameter optimisation, simulation-based optimisation, and experimental design, where evaluations are costly and uncertainty must be managed explicitly.

This project supports my current and future career by developing practical skills in Bayesian optimisation, uncertainty-aware modelling, and iterative decision-making. These skills are directly transferable to data science, ML engineering, and research roles where optimisation must be performed with incomplete knowledge.

---

## Section 2: Inputs and Outputs

Each iteration consists of submitting a single query point per function and receiving a scalar response.

### Inputs

- **Query format:** `x1-x2-x3-...-xn`
- Each `xi`:
  - Lies in the range `[0, 1]`
  - Is specified to **six decimal places**
- **Dimensionality:** varies by function (from 2D to 8D)
- **Constraint:** one query per function per iteration

**Example input (2D):**

```
0.372451-0.684219
```

### Outputs

- A single real-valued scalar representing the function response
- Output scale, smoothness, and noise level are unknown and function-specific
- Outputs are used only to inform subsequent modelling and query decisions

---

## Section 3: Challenge Objectives

The primary objective of the BBO capstone project is to **maximise the output of each unknown function** over successive iterations.

Key constraints include:

- A limited total number of allowable queries
- Sequential feedback (results are only available after submission)
- Unknown function structure and noise characteristics
- Increasing dimensionality, which increases sparsity and uncertainty

The challenge therefore requires careful trade-offs between learning about the function landscape and exploiting currently promising regions.

---

## Section 4: Technical Approach

# 📊 Architecture Diagram

            +----------------------+
            |   Historical Data    |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Weekly Inputs/Outputs|
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Gaussian Process     |
            |  (Kernel Selection)  |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Acquisition Function |
            | (EI / UCB with ξ, β) |
            +----------+-----------+
                       |
                       v
            +----------------------+
            | Next Query Generator |
            +----------+-----------+
                       |
                       v
            +----------------------+
            |   BBO Portal Output  |
            +----------------------+

---

# 📂 Repository Structure

bbo-project/
│
├── bbo_gp_weekly_generator.py
├── plots/
│ ├── func01/
│ ├── func02/
│ └── ...
├── history/
├── README.md

---

### Strategy Evolution (Weeks 1–n)

- **Week 1 – Structured Exploration:**
  Initial queries prioritised broad exploration and diversity. With no prior feedback available, points were chosen to reduce global uncertainty and avoid boundary bias, particularly for higher-dimensional functions.

- **Week 2 – Adaptive Exploration vs Exploitation:**
  Observed outputs from Week 1 enabled relative performance comparisons. Most functions remained exploratory due to high uncertainty, while a small number were refined locally based on promising results.

- **Week 3 – Model-Driven Bayesian Optimisation:**
  A separate Gaussian Process (GP) model was fitted for each function. Kernel choice and hyperparameters were optimised automatically by maximising the log marginal likelihood. Query points were selected using acquisition functions informed by the posterior mean μ(x) and predictive uncertainty σ(x).

- **Week 4 Update By Week 4, the optimisation strategy evolved into a **fully model-driven Bayesian optimisation pipeline\*\*, with:
- Automatic kernel selection using log-marginal likelihood
- Per-function exploration/exploitation tuning
- Diagnostics tracking kernel, ξ, β, and exploration ratio

This reflects real-world ML workflows where optimisation must adapt as more data becomes available.

### Methods Used

- Gaussian Process regression with RBF and Matérn kernels
- Automatic kernel and hyperparameter selection via marginal likelihood
- Expected Improvement (EI) as the primary acquisition function
- Explicit exploration–exploitation control using:
  - **ξ (xi)** for EI (e.g. ξ = 0.001 for exploitation, ξ = 0.05 for exploration)
  - **β (beta)** for UCB-style acquisition when applicable

### Exploration vs Exploitation

Exploration and exploitation are balanced explicitly through acquisition parameters rather than fixed heuristics. Lower ξ (or β) biases the search toward high-performing regions, while higher values encourage sampling uncertain areas. This enables function-specific strategies that adapt as more data becomes available.

### Future Extensions

Support Vector Machines (SVMs) could be used to classify regions as high or low performance using a soft-margin formulation. Kernel SVMs would be particularly useful for non-linear decision boundaries. While SVMs lack calibrated uncertainty, they could complement GP-based optimisation by identifying promising regions or decision boundaries.

---

This README reflects my current understanding and approach and will continue to evolve as additional iterations and modelling strategies are explored throughout the BBO capstone project.

### Data-Science Learning

This black-box setup teaches real-world ML thinking:

- Work with incomplete knowledge
- Model uncertainty explicitly
- Iterate based on feedback
- Justify decisions with diagnostics

These are critical skills for ML engineering and risk analytics projects.

This project demonstrates ability to:

- Design optimisation pipelines
- Apply Bayesian optimisation
- Build scalable ML systems
- Work under uncertainty
- Translate theory into production-ready Python code

---
