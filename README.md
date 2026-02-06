# Black-Box Optimisation (BBO) Capstone Project

## 1. Project overview

This repository contains my work for the **Black-Box Optimisation (BBO) capstone project**, where the objective is to optimise a set of unknown functions by iteratively querying them and observing their outputs. The internal form of each function is hidden, and only limited feedback is provided after each submission round.

The overall goal of the BBO capstone project is to **efficiently maximise the output of each unknown function under a strict query budget**. This setting closely reflects real-world machine learning problems such as hyperparameter tuning, simulation-based optimisation, and experimental design, where function evaluations are expensive and the response surface is unknown.

This project supports my current and future career by developing practical skills in **Bayesian optimisation, uncertainty-aware modelling, and decision-making under incomplete information**—all of which are directly applicable to applied ML, data science, and solution architecture roles.

---

## 2. Inputs and outputs

Each black-box function receives **one query input per week**, with dimensionality varying across functions.

### Inputs
- **Format:** `x1-x2-x3-...-xn`
- **Domain:** each `xi ∈ [0, 1]`
- **Precision:** six decimal places per value
- **Dimensionality:** varies by function (2D to 8D)
- **Constraint:** one query per function per round

**Examples**
- 2D input:  
  `0.372451-0.684219`
- 5D input:  
  `0.238415-0.564738-0.792164-0.413826-0.689541`

### Outputs
- A single real-valued scalar returned by the black-box function
- Represents a **performance signal** to be maximised
- Scale, smoothness, and noise level are unknown a priori

---

## 3. Challenge objectives

The objective of the BBO capstone project is to **maximise the output of each unknown function** using as few queries as possible.

Key constraints include:
- A **limited query budget** (one query per function per week)
- **Delayed feedback** (outputs are only available after submission)
- **Unknown function structure**, dimensionality, and noise characteristics
- High-dimensional search spaces with sparse observations

The challenge is therefore not only to find high-performing points, but to do so **efficiently**, balancing learning about the function with exploiting known promising regions.

---

## 4. Technical approach

This section is a living record of how my approach has evolved across the first three query rounds.

### Modelling
Each function is modelled using a **Gaussian Process (GP)** surrogate. GPs provide both a predictive mean (μ) and predictive uncertainty (σ), which are essential for principled exploration. Kernel choice (RBF, Matérn ν=1.5, Matérn ν=2.5), length-scales, and noise levels are selected automatically by maximising the **log marginal likelihood**, allowing the model to adapt to observed data.

### Query selection
New queries are generated using the **Expected Improvement (EI)** acquisition function. The balance between exploration and exploitation is controlled explicitly through the parameter **ξ (xi)**:
- **Exploitation-focused:** `ξ = 0.001`
- **Exploration-focused:** `ξ = 0.05`

In some experiments, an Upper Confidence Bound (UCB) formulation is also considered, where **β (beta)** controls the weight of uncertainty:
