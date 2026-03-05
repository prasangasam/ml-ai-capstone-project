#!/usr/bin/env python3
"""
Progressive visualization script to show BBO optimization progress at each week.
Creates separate folders with graphs for each week showing cumulative progress.
"""

import sys
sys.path.insert(0, str(__file__).replace('scripts\\progressive_visualize.py', 'src'))

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from bbo.data_loader import load_weekly_matrix, load_initial_from_dir

# Set style for better looking plots
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True

def load_data():
    """Load initial data and weekly progress data."""
    initial_dir = Path("data/initial_data")
    weekly_dir = Path("data/weekly")
    
    # Load initial data
    initial_data = load_initial_from_dir(initial_dir)
    
    # Load weekly data
    weekly_data = load_weekly_matrix(weekly_dir)
    if weekly_data is None:
        raise ValueError("Could not load weekly data")
    
    weekly_inputs, weekly_outputs = weekly_data
    
    return initial_data, weekly_inputs, weekly_outputs

def plot_function_progress_up_to_week(initial_data, weekly_outputs, up_to_week):
    """Plot the progress of each function up to a specific week."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    # Limit data to the specified week
    weekly_outputs_subset = weekly_outputs[:up_to_week]
    
    for func_idx in range(8):
        ax = axes[func_idx]
        
        # Get initial best value for this function
        initial_best = np.max(initial_data[func_idx].y)
        
        # Get weekly values for this function up to current week
        weekly_values = [week_outputs[func_idx] for week_outputs in weekly_outputs_subset]
        
        # Plot initial best as horizontal line
        ax.axhline(y=initial_best, color='red', linestyle='--', alpha=0.7, 
                   label=f'Initial Best: {initial_best:.3f}')
        
        # Plot weekly progress
        weeks = list(range(1, len(weekly_values) + 1))
        ax.plot(weeks, weekly_values, 'o-', linewidth=2, markersize=8,
                label='Weekly Results')
        
        # Highlight best result so far
        if weekly_values:
            best_so_far = np.max(weekly_values)
            best_week = np.argmax(weekly_values) + 1
            ax.scatter(best_week, best_so_far, color='gold', s=100, 
                       label=f'Best So Far: {best_so_far:.3f}')
        
        ax.set_title(f'Function {func_idx + 1} Progress (Through Week {up_to_week})')
        ax.set_xlabel('Week')
        ax.set_ylabel('Function Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if weekly_values:
            best_weekly = np.max(weekly_values)
            improvement = best_weekly - initial_best
            color = 'green' if improvement > 0 else 'red'
            ax.text(0.02, 0.98, f'Improvement: {improvement:+.3f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        # Set consistent y-axis limits based on all data
        all_weekly_values = [week_outputs[func_idx] for week_outputs in weekly_outputs]
        all_values = [initial_best] + all_weekly_values
        y_min, y_max = min(all_values), max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    plt.tight_layout()
    plt.suptitle(f'Function Optimization Progress Through Week {up_to_week}', 
                 y=1.02, fontsize=16)
    return fig

def plot_week_to_week_improvements_up_to_week(weekly_outputs, up_to_week):
    """Plot week-to-week improvements up to a specific week."""
    if up_to_week < 2:  # Need at least 2 weeks to show improvements
        return None
        
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    # Limit data to the specified week
    weekly_outputs_subset = weekly_outputs[:up_to_week]
    
    for func_idx in range(8):
        ax = axes[func_idx]
        
        # Calculate week-to-week differences
        weekly_values = [week_outputs[func_idx] for week_outputs in weekly_outputs_subset]
        improvements = [weekly_values[i] - weekly_values[i-1] for i in range(1, len(weekly_values))]
        
        if not improvements:
            ax.text(0.5, 0.5, 'No improvements data\n(Week 1)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Function {func_idx + 1} Week-to-Week Changes')
            continue
        
        # Plot improvements as bar chart
        weeks = list(range(2, len(weekly_values) + 1))  # Start from week 2
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.bar(weeks, improvements, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                    f'{imp:+.3f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title(f'Function {func_idx + 1} Week-to-Week Changes (Through Week {up_to_week})')
        ax.set_xlabel('Week')
        ax.set_ylabel('Improvement from Previous Week')
        ax.grid(True, alpha=0.3)
        
        # Add summary statistics
        total_improvement = sum(improvements)
        avg_improvement = np.mean(improvements)
        ax.text(0.02, 0.98, f'Total: {total_improvement:+.3f}\nAvg: {avg_improvement:+.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Set consistent y-axis limits
        all_improvements = []
        for week_idx in range(1, len(weekly_outputs)):
            all_weekly_values = [week_outputs[func_idx] for week_outputs in weekly_outputs[:week_idx+1]]
            if len(all_weekly_values) >= 2:
                imp = all_weekly_values[-1] - all_weekly_values[-2]
                all_improvements.append(imp)
        
        if all_improvements:
            y_min, y_max = min(all_improvements), max(all_improvements)
            y_range = max(y_max - y_min, 0.1)
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    plt.tight_layout()
    plt.suptitle(f'Week-to-Week Improvements Through Week {up_to_week}', 
                 y=1.02, fontsize=16)
    return fig

def plot_overall_summary_up_to_week(initial_data, weekly_outputs, up_to_week):
    """Plot overall performance summary up to a specific week."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Limit data to the specified week
    weekly_outputs_subset = weekly_outputs[:up_to_week]
    
    # 1. Best value per week across all functions
    weekly_bests = []
    weekly_means = []
    for week_outputs in weekly_outputs_subset:
        weekly_bests.append(np.max(week_outputs))
        weekly_means.append(np.mean(week_outputs))
    
    # Add initial best to comparison
    initial_bests = [np.max(func.y) for func in initial_data]
    initial_best_overall = np.max(initial_bests)
    initial_mean_overall = np.mean(initial_bests)
    
    weeks = list(range(1, len(weekly_bests) + 1))
    
    ax1.plot([0] + weeks, [initial_best_overall] + weekly_bests, 'o-', 
             linewidth=2, markersize=8, label='Best Value')
    ax1.plot([0] + weeks, [initial_mean_overall] + weekly_means, 's-', 
             linewidth=2, markersize=8, label='Mean Value')
    ax1.set_title(f'Performance Trajectory (Through Week {up_to_week})')
    ax1.set_xlabel('Week (0 = Initial)')
    ax1.set_ylabel('Function Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of improvements
    all_improvements = []
    for func_idx in range(8):
        if weekly_outputs_subset:
            weekly_values = [week_outputs[func_idx] for week_outputs in weekly_outputs_subset]
            best_weekly = np.max(weekly_values)
            initial_best = np.max(initial_data[func_idx].y)
            improvement = best_weekly - initial_best
        else:
            improvement = 0
        all_improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in all_improvements]
    bars = ax2.bar(range(1, 9), all_improvements, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, all_improvements)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.05),
                f'{imp:+.2f}', ha='center', va='bottom' if height > 0 else 'top')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title(f'Improvement by Function (Through Week {up_to_week})')
    ax2.set_xlabel('Function')
    ax2.set_ylabel('Improvement (Best So Far - Initial Best)')
    ax2.set_xticks(range(1, 9))
    ax2.grid(True, alpha=0.3)
    
    # 3. Success rate and statistics
    successful_functions = sum(1 for imp in all_improvements if imp > 0)
    success_rate = successful_functions / 8 * 100
    
    stats_data = [successful_functions, 8 - successful_functions]
    labels = [f'Improved ({successful_functions})', f'No Improvement ({8 - successful_functions})']
    colors_pie = ['green', 'red']
    
    wedges, texts, autotexts = ax3.pie(stats_data, labels=labels, colors=colors_pie, 
                                       autopct='%1.0f%%', startangle=90)
    ax3.set_title(f'Success Rate: {success_rate:.0f}%\n'
                  f'Avg Improvement: {np.mean(all_improvements):+.3f}\n'
                  f'(Through Week {up_to_week})')
    
    plt.tight_layout()
    plt.suptitle(f'BBO Performance Summary Through Week {up_to_week}', y=1.02, fontsize=16)
    return fig

def plot_heatmap_up_to_week(weekly_outputs, up_to_week):
    """Create a heatmap showing the optimization landscape up to a specific week."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Limit data to the specified week
    weekly_outputs_subset = weekly_outputs[:up_to_week]
    
    if not weekly_outputs_subset:
        ax.text(0.5, 0.5, f'No data available for Week {up_to_week}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Function Values Through Week {up_to_week}')
        return fig
    
    # Create matrix of weekly outputs
    output_matrix = np.array(weekly_outputs_subset).T  # Functions x Weeks
    
    # Create heatmap
    im = ax.imshow(output_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Function Value', rotation=270, labelpad=15)
    
    # Customize axes
    ax.set_xticks(range(len(weekly_outputs_subset)))
    ax.set_xticklabels([f'Week {i+1}' for i in range(len(weekly_outputs_subset))])
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Function {i+1}' for i in range(8)])
    
    # Add text annotations
    for i in range(8):
        for j in range(len(weekly_outputs_subset)):
            text = ax.text(j, i, f'{output_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'Function Values Heatmap (Through Week {up_to_week})')
    ax.set_xlabel('Week')
    ax.set_ylabel('Function')
    
    return fig

def generate_progressive_visualizations():
    """Generate progressive visualizations for each week."""
    print("Loading data...")
    initial_data, weekly_inputs, weekly_outputs = load_data()
    
    total_weeks = len(weekly_outputs)
    print(f"Generating visualizations for {total_weeks} weeks...")
    
    # Create main output directory
    output_dir = Path("artifacts/progressive_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for week in range(1, total_weeks + 1):
        print(f"\nGenerating visualizations for Week {week}...")
        
        # Create week-specific directory
        week_dir = output_dir / f"week{week}"
        week_dir.mkdir(exist_ok=True)
        
        # Generate function progress plot
        fig1 = plot_function_progress_up_to_week(initial_data, weekly_outputs, week)
        fig1.savefig(week_dir / f"week{week}_function_progress.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Generate week-to-week improvements plot (only if week >= 2)
        if week >= 2:
            fig2 = plot_week_to_week_improvements_up_to_week(weekly_outputs, week)
            if fig2:
                fig2.savefig(week_dir / f"week{week}_improvements.png", dpi=300, bbox_inches='tight')
                plt.close(fig2)
        
        # Generate overall summary plot
        fig3 = plot_overall_summary_up_to_week(initial_data, weekly_outputs, week)
        fig3.savefig(week_dir / f"week{week}_summary.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # Generate heatmap
        fig4 = plot_heatmap_up_to_week(weekly_outputs, week)
        fig4.savefig(week_dir / f"week{week}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        print(f"  ✓ Week {week} visualizations saved to: {week_dir}")
    
    print(f"\n🎉 All progressive visualizations completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Generated {total_weeks} week folders with historical progress graphs")
    
    # Generate summary
    print(f"\n📋 Summary:")
    for week in range(1, total_weeks + 1):
        week_dir = output_dir / f"week{week}"
        files = list(week_dir.glob("*.png"))
        print(f"   Week {week}: {len(files)} visualization files")

if __name__ == "__main__":
    generate_progressive_visualizations()