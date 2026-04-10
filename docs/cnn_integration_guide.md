# CNN Integration for Black-Box Optimization (BBO)

This document explains how Convolutional Neural Networks (CNNs) can be integrated into the BBO optimization pipeline to enhance performance and provide alternative approaches to Gaussian Process (GP) surrogate modeling.

---

# Using CNN with Your Existing BBO Code

## Quick Start: Three Ways to Use CNN

### Option 1: Replace Standard Pipeline (Easiest)

Replace your current optimization run with the CNN-enhanced version:

```python
# Current standard run:
# python scripts/run_week.py --initial_dir data/initial_data --weekly_dir data/weekly

# CNN-enhanced run:
from pathlib import Path
from bbo.cnn_integration import run_cnn_enhanced_optimization

result = run_cnn_enhanced_optimization(
    initial_dir=Path("data/initial_data"),
    weekly_dir=Path("data/weekly"),
    use_hybrid=True,
    cnn_ensemble_weights={"gp": 0.7, "cnn_surrogate": 0.3}
)

print(f"Week {result['week_k']} completed with CNN enhancement")
```

### Option 2: Individual Function Replacement

Replace GP with CNN for specific functions in your existing pipeline:

```python
# In your existing pipeline.py run() function:
from bbo.cnn_surrogate import propose_next_point_cnn

# Standard GP approach (existing)
x_next_gp, report_gp = propose_next_point(
    f.X, f.y,
    acquisition=config.ACQUISITION,
    xi=xi, beta=beta,
    seed=config.RNG_SEED + 31*i,
    n_candidates=config.N_CANDIDATES,
)

# CNN alternative (for 4D+ functions with 25+ points)
if f.X.shape[1] >= 4 and len(f.y) >= 25:
    x_next, report = propose_next_point_cnn(
        f.X, f.y,
        acquisition=config.ACQUISITION,
        xi=xi,
        seed=config.RNG_SEED + 31*i,
        n_candidates=config.N_CANDIDATES,
    )
    print(f"Function {i}: Using CNN surrogate model")
else:
    x_next, report = x_next_gp, report_gp
    print(f"Function {i}: Using GP model")
```

### Option 3: Week 9 Integration (Advanced Users)

Combine CNN with the current Week 8 and Week 9 strategy controls:

```python
from bbo.strategy import adaptive_exploration_params, choose_strategy, recent_instability, emergence_score
from bbo.cnn_integration import HybridCNNGPOptimizer


def week9_cnn_optimization(funcs, week_k=9):
    optimizer = HybridCNNGPOptimizer({
        "gp": 0.6,
        "cnn_surrogate": 0.4
    })

    results = []
    for i, f in enumerate(funcs):
        strategy = choose_strategy(week_k, f.y, dim=f.X.shape[1])
        advanced_params = adaptive_exploration_params(week_k, f.y, i, dim=f.X.shape[1])

        x_next, report = optimizer.propose_hybrid_point(
            f.X, f.y,
            acquisition="ucb" if strategy == "hedge" else "ei",
            xi=advanced_params["xi"],
            beta=advanced_params["beta"],
            func_idx=i
        )

        report["instability"] = recent_instability(f.y)
        report["emergence_score"] = emergence_score(f.y)
        results.append((x_next, report))

        print(f"Function {i+1}: {strategy} | emergence={report['emergence_score']:.3f}")

    return results
```

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
- As an ensemble complement when emergence or scaling pressure is high

### Basic Usage Example

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

## 2. Installation and Setup

### Prerequisites

```bash
# Install CNN dependencies
pip install torch torchvision

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Test CNN Integration

```python
# Test CNN with your existing data
import numpy as np
from pathlib import Path
from bbo.cnn_surrogate import CNNBayesianOptimizer

# Load your actual function data
from bbo.data_loader import load_initial_from_dir, load_weekly
seeds = load_initial_from_dir(Path("data/initial_data"))
weekly_inputs, weekly_outputs, _ = load_weekly(Path("data/weekly"))

# Test on Function 4 (4D, good for CNN)
func_4_data = seeds[3]  # Function 4 (0-indexed)
X, y = func_4_data.X, func_4_data.y

# Add weekly data
for week_in, week_out in zip(weekly_inputs, weekly_outputs):
    X = np.vstack([X, week_in[3].reshape(1, -1)])
    y = np.append(y, week_out[3])

print(f"Testing CNN on Function 4: {len(y)} points in {X.shape[1]}D")

# Train CNN
cnn_opt = CNNBayesianOptimizer(input_dim=X.shape[1])
cnn_opt.config.epochs = 50  # Adjust for your needs
fit_info = cnn_opt.fit(X, y)

print(f"CNN training completed, loss: {fit_info['final_loss']:.6f}")
```

## 3. Integration Patterns

### Pattern A: Drop-in Replacement

Replace GP calls directly in your existing pipeline:

```python
# In pipeline.py, replace this:
x_next, report = propose_next_point(
    f.X, f.y,
    acquisition=config.ACQUISITION,
    xi=xi, beta=beta,
    seed=config.RNG_SEED + 31*i,
    n_candidates=config.N_CANDIDATES,
)

# With this (for suitable functions):
from bbo.cnn_surrogate import propose_next_point_cnn

if should_use_cnn(f.X.shape[1], len(f.y)):
    x_next, report = propose_next_point_cnn(
        f.X, f.y,
        acquisition=config.ACQUISITION,
        xi=xi,
        seed=config.RNG_SEED + 31*i,
        n_candidates=config.N_CANDIDATES,
    )
else:
    x_next, report = propose_next_point(...)  # Standard GP
```

### Pattern B: Ensemble Method

Combine both approaches:

```python
from bbo.cnn_integration import HybridCNNGPOptimizer

# Create ensemble optimizer
optimizer = HybridCNNGPOptimizer({
    "gp": 0.7,             # 70% weight to GP
    "cnn_surrogate": 0.3  # 30% weight to CNN
})

# Use in your pipeline
for i, f in enumerate(funcs):
    x_next, report = optimizer.propose_hybrid_point(
        f.X, f.y,
        acquisition=config.ACQUISITION,
        xi=xi, beta=beta,
        func_idx=i
    )

    print(f"Function {i}: Models used: {report['ensemble_info']['used_models']}")
```

### Pattern C: Conditional Usage

Use CNN only when conditions are met:

```python
def choose_optimization_method(X, y, func_idx):
    dim = X.shape[1]
    n_points = len(y)

    # Decision logic
    if dim >= 4 and n_points >= 25:
        return "cnn_surrogate"
    elif dim == 2 and n_points >= 15:
        return "cnn_landscape"  # For 2D visualization
    else:
        return "gp"  # Default fallback
```

## 4. Recommended Usage with the Current Repository

The current repository centres on Gaussian Processes, with Week 8 and Week 9 enhancements focused on instability handling, emergence detection, ruggedness control, and dimension-aware strategy switching. Within that design, CNN integration is most useful in the following roles:

- **High-dimensional surrogate support:** Use CNN models when function dimensionality is high and historical data volume begins to exceed the comfortable range for standard GP fitting.
- **Ensemble robustness:** Blend CNN and GP predictions when the latest observations suggest emergent behaviour or a rugged response surface.
- **Post-hoc landscape analysis:** Use CNN-based models to inspect candidate regions discovered by the hedge strategy rather than replacing the main weekly submission pipeline immediately.
- **Experimental comparison:** Benchmark CNN-guided points against GP-guided points on the same historical dataset before promoting the CNN path into the main capstone submission workflow.

## 5. Practical Notes

- The Week 9 pipeline already includes a fast GP fallback for the most demanding late-stage cases, so CNN integration should be treated as an optional enhancement rather than a required replacement.
- If weekly outputs are stored as NumPy scalar wrappers such as `np.float64(...)`, the hardened data loader will normalise them before training either GP or CNN models.
- For reproducible capstone submissions, keep the final portal formatting step inside the existing `io.py` and pipeline workflow even if candidate generation is delegated to CNN utilities.

---

This guide preserves the existing project style while aligning CNN usage recommendations with the current Week 9 scaling and emergence-aware optimisation framework.
