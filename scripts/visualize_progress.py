#!/usr/bin/env python3
"""
Visualization script to show weekly progress and improvements in BBO optimization.
"""

import sys
sys.path.insert(0, str(__file__).replace('scripts\\visualize_progress.py', 'src'))

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

def plot_function_progress(initial_data, weekly_outputs):
    """Plot the progress of each function over weeks."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for func_idx in range(8):
        ax = axes[func_idx]
        
        # Get initial best value for this function
        initial_best = np.max(initial_data[func_idx].y)
        
        # Get weekly values for this function
        weekly_values = [week_outputs[func_idx] for week_outputs in weekly_outputs]
        
        # Plot initial best as horizontal line
        ax.axhline(y=initial_best, color='red', linestyle='--', alpha=0.7, 
                   label=f'Initial Best: {initial_best:.3f}')
        
        # Plot weekly progress
        weeks = list(range(1, len(weekly_values) + 1))
        ax.plot(weeks, weekly_values, 'o-', linewidth=2, markersize=8,
                label='Weekly Results')
        
        # Highlight best weekly result
        best_weekly = np.max(weekly_values)
        best_week = np.argmax(weekly_values) + 1
        ax.scatter(best_week, best_weekly, color='gold', s=100, 
                   label=f'Best Weekly: {best_weekly:.3f}')
        
        ax.set_title(f'Function {func_idx + 1} Progress')
        ax.set_xlabel('Week')
        ax.set_ylabel('Function Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotation
        improvement = best_weekly - initial_best
        color = 'green' if improvement > 0 else 'red'
        ax.text(0.02, 0.98, f'Improvement: {improvement:+.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    plt.suptitle('Function Optimization Progress Over Weeks', y=1.02, fontsize=16)
    return fig

def plot_week_to_week_improvements(weekly_outputs):
    """Plot week-to-week improvements for each function."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for func_idx in range(8):
        ax = axes[func_idx]
        
        # Calculate week-to-week differences
        weekly_values = [week_outputs[func_idx] for week_outputs in weekly_outputs]
        improvements = [weekly_values[i] - weekly_values[i-1] for i in range(1, len(weekly_values))]
        
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
        ax.set_title(f'Function {func_idx + 1} Week-to-Week Changes')
        ax.set_xlabel('Week')
        ax.set_ylabel('Improvement from Previous Week')
        ax.grid(True, alpha=0.3)
        
        # Add summary statistics
        total_improvement = sum(improvements)
        avg_improvement = np.mean(improvements)
        ax.text(0.02, 0.98, f'Total: {total_improvement:+.3f}\nAvg: {avg_improvement:+.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.suptitle('Week-to-Week Improvements by Function', y=1.02, fontsize=16)
    return fig

def plot_overall_performance_summary(initial_data, weekly_outputs):
    """Plot overall performance summary across all functions."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Best value per week across all functions
    weekly_bests = []
    weekly_means = []
    for week_outputs in weekly_outputs:
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
    ax1.set_title('Overall Performance Trajectory')
    ax1.set_xlabel('Week (0 = Initial)')
    ax1.set_ylabel('Function Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of improvements
    all_improvements = []
    for func_idx in range(8):
        weekly_values = [week_outputs[func_idx] for week_outputs in weekly_outputs]
        best_weekly = np.max(weekly_values)
        initial_best = np.max(initial_data[func_idx].y)
        improvement = best_weekly - initial_best
        all_improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in all_improvements]
    bars = ax2.bar(range(1, 9), all_improvements, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, all_improvements)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.05),
                f'{imp:+.2f}', ha='center', va='bottom' if height > 0 else 'top')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Total Improvement by Function')
    ax2.set_xlabel('Function')
    ax2.set_ylabel('Improvement (Best Weekly - Initial Best)')
    ax2.set_xticks(range(1, 9))
    ax2.grid(True, alpha=0.3)
    
    # 3. Success rate and statistics
    successful_functions = sum(1 for imp in all_improvements if imp > 0)
    success_rate = successful_functions / 8 * 100
    
    stats_data = [
        successful_functions,
        8 - successful_functions,
    ]
    labels = [f'Improved ({successful_functions})', f'No Improvement ({8 - successful_functions})']
    colors_pie = ['green', 'red']
    
    wedges, texts, autotexts = ax3.pie(stats_data, labels=labels, colors=colors_pie, 
                                       autopct='%1.0f%%', startangle=90)
    ax3.set_title(f'Success Rate: {success_rate:.0f}%\n'
                  f'Avg Improvement: {np.mean(all_improvements):+.3f}')
    
    plt.tight_layout()
    plt.suptitle('Overall BBO Performance Summary', y=1.02, fontsize=16)
    return fig

def plot_exploration_vs_exploitation_heatmap(weekly_outputs):
    """Create a heatmap showing the optimization landscape."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create matrix of weekly outputs
    output_matrix = np.array(weekly_outputs).T  # Functions x Weeks
    
    # Create heatmap
    im = ax.imshow(output_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Function Value', rotation=270, labelpad=15)
    
    # Customize axes
    ax.set_xticks(range(len(weekly_outputs)))
    ax.set_xticklabels([f'Week {i+1}' for i in range(len(weekly_outputs))])
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Function {i+1}' for i in range(8)])
    
    # Add text annotations
    for i in range(8):
        for j in range(len(weekly_outputs)):
            text = ax.text(j, i, f'{output_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Function Values Across Weeks (Heatmap)')
    ax.set_xlabel('Week')
    ax.set_ylabel('Function')
    
    return fig

def save_all_plots():
    """Generate and save all visualization plots."""
    print("Loading data...")
    initial_data, weekly_inputs, weekly_outputs = load_data()
    
    # Create output directory
    output_dir = Path("artifacts/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating function progress plot...")
    fig1 = plot_function_progress(initial_data, weekly_outputs)
    fig1.savefig(output_dir / "function_progress.png", dpi=300, bbox_inches='tight')
    
    print("Generating week-to-week improvements plot...")
    fig2 = plot_week_to_week_improvements(weekly_outputs)
    fig2.savefig(output_dir / "week_to_week_improvements.png", dpi=300, bbox_inches='tight')
    
    print("Generating overall performance summary...")
    fig3 = plot_overall_performance_summary(initial_data, weekly_outputs)
    fig3.savefig(output_dir / "overall_performance_summary.png", dpi=300, bbox_inches='tight')
    
    print("Generating optimization heatmap...")
    fig4 = plot_exploration_vs_exploitation_heatmap(weekly_outputs)
    fig4.savefig(output_dir / "optimization_heatmap.png", dpi=300, bbox_inches='tight')
    
    plt.show()  # Display all plots
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated plots:")
    print("1. function_progress.png - Weekly progress for each function")
    print("2. week_to_week_improvements.png - Week-to-week changes")
    print("3. overall_performance_summary.png - Overall performance metrics")
    print("4. optimization_heatmap.png - Function values heatmap")

if __name__ == "__main__":
    save_all_plots()