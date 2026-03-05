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

### Option 3: Week 6 Integration (Advanced Users)

Combine CNN with existing Week 6 advanced features:

```python
# Enhanced Week 6 optimization with CNN
from bbo.strategy import adaptive_exploration_params
from bbo.cnn_integration import HybridCNNGPOptimizer

def week6_cnn_optimization(funcs, week_k=6):
    optimizer = HybridCNNGPOptimizer({
        "gp": 0.6,
        "cnn_surrogate": 0.4
    })
    
    for i, f in enumerate(funcs):
        # Week 6 advanced parameters
        advanced_params = adaptive_exploration_params(week_k, f.y, i)
        
        # CNN-enhanced optimization
        x_next, report = optimizer.propose_hybrid_point(
            f.X, f.y,
            acquisition="ei",
            xi=advanced_params["xi"],
            beta=advanced_params["beta"],
            func_idx=i
        )
        
        print(f"Function {i+1}: {report['ensemble_info']['used_models']}")
    
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
    "gp": 0.7,           # 70% weight to GP
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

# In your optimization loop:
method = choose_optimization_method(f.X, f.y, i)

if method == "cnn_surrogate":
    x_next, report = propose_next_point_cnn(f.X, f.y, ...)
elif method == "gp":
    x_next, report = propose_next_point(f.X, f.y, ...)
```

## 4. Practical Examples

### Example 1: Modify Existing run_week.py

Create a CNN-enhanced version:

```python
#!/usr/bin/env python3
"""Enhanced run_week.py with CNN integration"""

from pathlib import Path
import argparse
from bbo.cnn_integration import run_cnn_enhanced_optimization

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_dir', type=Path, required=True)
    parser.add_argument('--weekly_dir', type=Path, required=True)
    parser.add_argument('--use_cnn', action='store_true', help='Enable CNN enhancement')
    parser.add_argument('--cnn_weight', type=float, default=0.3, help='CNN ensemble weight')
    
    args = parser.parse_args()
    
    if args.use_cnn:
        # CNN-enhanced optimization
        ensemble_weights = {
            "gp": 1.0 - args.cnn_weight,
            "cnn_surrogate": args.cnn_weight
        }
        
        result = run_cnn_enhanced_optimization(
            initial_dir=args.initial_dir,
            weekly_dir=args.weekly_dir,
            use_hybrid=True,
            cnn_ensemble_weights=ensemble_weights
        )
        
        print(f"🤖 CNN-Enhanced Week {result['week_k']} completed")
        print(f"📊 Models used: {result['diagnostics_summary']['unique_models']}")
    else:
        # Standard optimization (your existing code)
        from bbo.pipeline import run
        result = run(initial_dir=args.initial_dir, weekly_dir=args.weekly_dir)
        print(f"📊 Standard Week {result['week_k']} completed")
    
    print(f"💾 Results saved to: {result['submission_path']}")

if __name__ == "__main__":
    main()
```

Usage:
```bash
# Standard optimization
python scripts/run_week_enhanced.py --initial_dir data/initial_data --weekly_dir data/weekly

# CNN-enhanced optimization  
python scripts/run_week_enhanced.py --initial_dir data/initial_data --weekly_dir data/weekly --use_cnn --cnn_weight 0.4
```

### Example 2: Function-Specific CNN Usage

Customize CNN usage per function:

```python
# Configure CNN usage per function
CNN_CONFIG = {
    1: {"use_cnn": False, "reason": "2D function, GP sufficient"},
    2: {"use_cnn": False, "reason": "2D function, GP sufficient"}, 
    3: {"use_cnn": False, "reason": "3D function, GP sufficient"},
    4: {"use_cnn": True, "reason": "4D function, CNN beneficial"},
    5: {"use_cnn": True, "reason": "4D function, CNN beneficial"},
    6: {"use_cnn": True, "reason": "5D function, CNN beneficial"},
    7: {"use_cnn": True, "reason": "6D function, CNN beneficial"},
    8: {"use_cnn": True, "reason": "8D function, CNN optimal"},
}

# In your optimization loop:
for i, f in enumerate(funcs, start=1):
    config = CNN_CONFIG[i]
    
    if config["use_cnn"] and len(f.y) >= 25:
        # Use CNN for high-dimensional functions with sufficient data
        x_next, report = propose_next_point_cnn(f.X, f.y, ...)
        print(f"Function {i}: CNN - {config['reason']}")
    else:
        # Use standard GP
        x_next, report = propose_next_point(f.X, f.y, ...)
        print(f"Function {i}: GP - {config['reason']}")
```

## 5. Monitoring and Debugging

### Performance Comparison

```python
# Compare GP vs CNN performance
def compare_methods(X, y):
    from bbo.gp import propose_next_point
    from bbo.cnn_surrogate import propose_next_point_cnn
    import time
    
    # Test GP
    start = time.time()
    x_gp, report_gp = propose_next_point(X, y, acquisition="ei", xi=0.01, beta=1.0, seed=42, n_candidates=1000)
    gp_time = time.time() - start
    
    # Test CNN
    start = time.time()
    x_cnn, report_cnn = propose_next_point_cnn(X, y, acquisition="ei", xi=0.01, seed=42, n_candidates=1000)
    cnn_time = time.time() - start
    
    print(f"GP Time: {gp_time:.2f}s, CNN Time: {cnn_time:.2f}s")
    print(f"GP Point: {x_gp}, CNN Point: {x_cnn}")
    
    return {"gp": (x_gp, report_gp, gp_time), "cnn": (x_cnn, report_cnn, cnn_time)}
```

### Error Handling

```python
# Robust CNN integration with fallback
def optimize_with_cnn_fallback(X, y, **kwargs):
    try:
        # Try CNN first
        x_next, report = propose_next_point_cnn(X, y, **kwargs)
        report["method_used"] = "cnn_surrogate"
        return x_next, report
        
    except Exception as e:
        print(f"⚠️ CNN optimization failed: {e}")
        print("🔄 Falling back to GP optimization...")
        
        # Fallback to GP
        x_next, report = propose_next_point(X, y, **kwargs)
        report["method_used"] = "gp_fallback"
        report["cnn_error"] = str(e)
        return x_next, report
```

## 6. Best Practices

### When to Use CNN vs GP

| Condition | Recommendation | Reason |
|-----------|---------------|---------|
| 2D-3D functions | Use GP | GP excellent for low dimensions |
| 4D+ functions with 25+ points | Use CNN | CNN scales better to high dimensions |
| Less than 20 evaluations | Use GP | CNN needs more training data |
| Week 1-3 | Use GP | Build initial dataset first |
| Week 4+ | Consider CNN | Sufficient data for CNN training |
| Week 6+ | Use hybrid | Combine advanced features |

### Performance Tips

```python
# Optimize CNN training for your use case
cnn_opt = CNNBayesianOptimizer(input_dim=dim)

# For faster training (testing)
cnn_opt.config.epochs = 20
cnn_opt.config.learning_rate = 0.01

# For better accuracy (production)
cnn_opt.config.epochs = 100
cnn_opt.config.learning_rate = 0.001
cnn_opt.config.uncertainty_samples = 100  # More uncertainty samples

# Train once, use many times
fit_info = cnn_opt.fit(X, y)
mu1, sigma1 = cnn_opt.predict_with_mc_uncertainty(X_test1)
mu2, sigma2 = cnn_opt.predict_with_mc_uncertainty(X_test2)
```

**Start with Option 1 (replace standard pipeline) for easiest integration, then experiment with Options 2-3 for more control over the CNN usage in your specific BBO problem.**
