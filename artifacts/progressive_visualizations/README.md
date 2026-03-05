# Progressive BBO Optimization Visualizations

This directory contains week-by-week visualizations showing the historical progression of the Bayesian Black-Box Optimization campaign.

## Directory Structure

Each `weekX` folder contains visualizations showing the optimization state **through that specific week**, preserving the historical view of progress at each point in time.

### Week Folders

- **week1/**: Progress through Week 1 only
- **week2/**: Progress through Week 2 (includes Week 1 data)
- **week3/**: Progress through Week 3 (includes Weeks 1-2 data)
- **week4/**: Progress through Week 4 (includes Weeks 1-3 data)
- **week5/**: Progress through Week 5 (includes all historical data)

### Visualization Files in Each Week Folder

#### For Week 1:
- `week1_function_progress.png`: Individual function optimization trajectories (Week 1 only)
- `week1_summary.png`: Overall performance summary and success rate (Week 1 only)
- `week1_heatmap.png`: Function values heatmap (Week 1 only)

#### For Week 2 and onwards:
- `weekX_function_progress.png`: Individual function optimization trajectories through Week X
- `weekX_improvements.png`: Week-to-week improvements bar charts through Week X
- `weekX_summary.png`: Overall performance summary and success rate through Week X
- `weekX_heatmap.png`: Function values heatmap through Week X

## How to Use

### To see the optimization state at any specific point in time:
- Open the corresponding `weekX` folder
- View the graphs to see how things looked at the end of that week

### To track progression over time:
- Compare the same graph type across different week folders
- Example: Compare `week1_summary.png`, `week2_summary.png`, `week3_summary.png` to see how the optimization campaign evolved

### Key Insights Available:

1. **Function-specific Performance**: Track how each of the 8 functions improved over time
2. **Week-to-Week Changes**: See which weeks had the biggest improvements or setbacks
3. **Overall Campaign Success**: Monitor success rate and average improvements as they accumulated
4. **Optimization Patterns**: Visualize the complete data landscape at any point in the campaign

## Generated on

- Date: March 5, 2026
- Total Weeks: 5
- Functions Optimized: 8
- BBO Strategy: Gaussian Process with Expected Improvement acquisition

---

*These visualizations preserve the complete historical record of your BBO optimization campaign, allowing you to see exactly how the optimization looked at any point in time.*