# CNN Integration for Black-Box Optimization (BBO)

This document explains how Convolutional Neural Networks (CNNs) can be integrated into the BBO optimization pipeline to enhance performance and provide alternative approaches to Gaussian Process (GP) surrogate modeling.

---

# CNN Approaches for BBO

## 1. CNN Surrogate Models (`cnn_surrogate.py`)

### Purpose

Replace or supplement Gaussian Processes with deep learning-based function approximation.

### Key Features

- Deep surrogate modeling with uncertainty quantification
- Monte Carlo dropout for uncertainty estimation
- Adaptive architecture based on function dimensionality
- Expected Improvement acquisition with CNN predictions

### When to Use

- High-dimensional functions (4D+)
- Large training datasets (25+ evaluations)
- Non-stationary or complex function landscapes
- When GP kernel assumptions may not hold

### Usage Example

```python
from bbo.cnn_surrogate import CNNBayesianOptimizer, propose_next_point_cnn

# Fit CNN surrogate model
X, y = function_evaluations
n_candidates = 10000

x_next, report = propose_next_point_cnn(
    X,
    y,
    acquisition="ei",
    xi=0.01,
    seed=42,
    n_candidates=n_candidates
)

print(f"Next point: {x_next}")
print(f"Model type: {report['model_type']}")
print(f"EI score: {report['ei_score']}")
```
