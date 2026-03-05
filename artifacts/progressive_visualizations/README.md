# Progressive BBO Optimization Visualizations

This directory contains week-by-week visualizations showing the historical progression of the Bayesian Black-Box Optimization campaign, including advanced optimization features and optional CNN-enhanced modeling.

## Directory Structure

Each `weekX` folder contains visualizations showing the optimization state **through that specific week**, preserving the historical view of progress at each point in time.

### Week Folders

- **week1/**: Progress through Week 1 - Structured exploration phase
- **week2/**: Progress through Week 2 - Adaptive exploration vs exploitation
- **week3/**: Progress through Week 3 - Model-driven Bayesian optimization
- **week4/**: Progress through Week 4 - Fully automated pipeline
- **week5/**: Progress through Week 5 - Progressive analysis integration
- **week6/**: Progress through Week 6 - **Advanced optimization and refinement** with sophisticated parameter tuning, convergence analysis, multi-objective portfolio balancing, and enhanced uncertainty quantification

### Visualization Files in Each Week Folder

#### For Week 1:
- `week1_function_progress.png`: Individual function optimization trajectories (Week 1 only)
- `week1_summary.png`: Overall performance summary and success rate (Week 1 only)
- `week1_heatmap.png`: Function values heatmap (Week 1 only)

#### For Week 2-5:
- `weekX_function_progress.png`: Individual function optimization trajectories through Week X
- `weekX_improvements.png`: Week-to-week improvements bar charts through Week X
- `weekX_summary.png`: Overall performance summary and success rate through Week X
- `weekX_heatmap.png`: Function values heatmap through Week X

#### For Week 6 (Advanced Optimization):
- `week6_function_progress.png`: Function optimization trajectories with convergence analysis
- `week6_improvements.png`: Week-to-week improvements with advanced parameter tuning
- `week6_summary.png`: Performance summary with portfolio balancing metrics
- `week6_heatmap.png`: Function values with enhanced uncertainty quantification
- `week6_convergence_analysis.png`: Convergence rates and stability scores
- `week6_portfolio_balancing.png`: Multi-objective function portfolio weights
- **Additional advanced visualizations available in week6/ folder**

## Advanced Features (Week 6+)

### Sophisticated Parameter Tuning
- **Adaptive exploration parameters** that decay over time (0.85^week factor)
- **Function-specific parameter adjustments** based on historical performance
- **Dynamic ξ/β tuning** responding to convergence state

### Convergence Analysis
- **Convergence rate monitoring** using 3-step sliding window analysis
- **Improvement trend detection** across recent optimization steps  
- **Stability scoring** based on performance variance

### Multi-Objective Portfolio Balancing
- **Function portfolio weighting** prioritizing underperforming functions
- **Balanced resource allocation** across 8 optimization targets
- **Dynamic rebalancing** based on relative performance

### Enhanced Uncertainty Quantification
- **Uncertainty boost factors** for unstable functions (1.5x multiplier)
- **Confidence-weighted decisions** in acquisition strategies
- **Robust decision-making** in final optimization phases

## Optional CNN Integration

For users with CNN capabilities installed (`torch`, `torchvision`), additional visualizations may include:

### CNN Surrogate Models (4D+ Functions)
- **Deep learning surrogate** model training progress
- **Monte Carlo uncertainty** visualization
- **CNN vs GP comparison** charts

### CNN Landscape Modeling (2D Functions)
- **Function landscape heatmaps** with spatial patterns
- **Learned acquisition function** visualizations
- **Spatial optimization** progress tracking

### Hybrid Ensemble Analysis
- **Model ensemble performance** comparisons
- **Adaptive model selection** tracking
- **CNN-GP hybrid optimization** effectiveness

## How to Use

### To see the optimization state at any specific point in time:
- Open the corresponding `weekX` folder
- View the graphs to see how things looked at the end of that week
- **Week 6** provides the most sophisticated analysis with advanced optimization metrics

### To track optimization strategy evolution:
- **Week 1-2**: Basic exploration and exploitation strategies
- **Week 3-4**: Gaussian Process modeling and automation  
- **Week 5**: Progressive analysis and visualization integration
- **Week 6**: Advanced optimization with convergence analysis, portfolio balancing, and enhanced uncertainty quantification

### Key Insights Available:

1. **Function-specific Performance**: Track how each of the 8 functions improved over time
2. **Advanced Parameter Evolution**: See how sophisticated parameter tuning adapted over weeks
3. **Convergence Patterns**: Monitor convergence rates and stability scores across functions
4. **Portfolio Optimization**: View multi-objective balancing strategies in action
5. **Uncertainty Analysis**: Track enhanced uncertainty quantification effectiveness
6. **Overall Campaign Success**: Monitor success rate improvements with advanced features

## Technical Implementation

### Standard Optimization (Weeks 1-5)
- Gaussian Process surrogate modeling
- Expected Improvement acquisition
- Basic exploration/exploitation balance

### Advanced Optimization (Week 6+)  
- Sophisticated parameter tuning with convergence analysis
- Multi-objective portfolio balancing across function portfolios
- Enhanced uncertainty quantification with adaptive boost factors
- Optional CNN-enhanced surrogate modeling and landscape optimization

## Generated on

- Date: March 5, 2026
- Total Weeks: 6 (including advanced optimization phase)
- Advanced Features: Active (Week 6+)
- CNN Integration: Available (optional dependency)
- Functions Optimized: 8
- BBO Strategy: Gaussian Process with Expected Improvement acquisition

---

*These visualizations preserve the complete historical record of your BBO optimization campaign, allowing you to see exactly how the optimization looked at any point in time.*