#!/usr/bin/env python3
"""
Visualization script to show weekly progress and improvements in BBO optimization.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
from bbo.data_loader import load_weekly_matrix, load_initial_from_dir

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True


def _to_scalar(value):
    """Convert nested arrays/lists/scalars into a plain float for plotting."""
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float('nan')
    return float(arr.reshape(-1)[0])


def _normalize_weekly_outputs(weekly_outputs):
    """Ensure weekly outputs are a list of lists of scalar floats."""
    return [[_to_scalar(v) for v in week] for week in weekly_outputs]


def safe_savefig(fig, path, dpi=300):
    """Save a figure with fallback when tight bbox creates an invalid canvas."""
    try:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    except Exception as exc:
        print(f"Warning: save with bbox_inches='tight' failed for {path.name}: {exc}")
        print("Retrying without tight bounding box...")
        fig.savefig(path, dpi=dpi)


def load_data():
    """Load initial data and weekly progress data."""
    initial_dir = PROJECT_ROOT / "data" / "initial_data"
    weekly_dir = PROJECT_ROOT / "data" / "weekly"

    initial_data = load_initial_from_dir(initial_dir)
    weekly_data = load_weekly_matrix(weekly_dir)
    if weekly_data is None:
        raise ValueError("Could not load weekly data")

    weekly_inputs, weekly_outputs = weekly_data
    weekly_outputs = _normalize_weekly_outputs(weekly_outputs)
    return initial_data, weekly_inputs, weekly_outputs


def _finalize_figure(fig, title):
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98, hspace=0.38, wspace=0.28)
    return fig


def plot_function_progress(initial_data, weekly_outputs):
    """Plot the progress of each function over weeks."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for func_idx in range(8):
        ax = axes[func_idx]
        initial_best = float(np.max(initial_data[func_idx].y))
        weekly_values = [float(week_outputs[func_idx]) for week_outputs in weekly_outputs]

        ax.axhline(y=initial_best, color='red', linestyle='--', alpha=0.7,
                   label=f'Initial Best: {initial_best:.3f}')
        weeks = list(range(1, len(weekly_values) + 1))
        ax.plot(weeks, weekly_values, 'o-', linewidth=2, markersize=8, label='Weekly Results')

        best_weekly = float(np.max(weekly_values))
        best_week = int(np.argmax(weekly_values)) + 1
        ax.scatter(best_week, best_weekly, color='gold', s=100, label=f'Best Weekly: {best_weekly:.3f}')

        ax.set_title(f'Function {func_idx + 1} Progress')
        ax.set_xlabel('Week')
        ax.set_ylabel('Function Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        improvement = best_weekly - initial_best
        color = 'green' if improvement > 0 else 'red'
        ax.text(
            0.02,
            0.98,
            f'Improvement: {improvement:+.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            fontsize=9,
        )

    return _finalize_figure(fig, 'Function Optimization Progress Over Weeks')


def plot_week_to_week_improvements(weekly_outputs):
    """Plot week-to-week improvements for each function."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for func_idx in range(8):
        ax = axes[func_idx]
        weekly_values = [float(week_outputs[func_idx]) for week_outputs in weekly_outputs]
        improvements = [weekly_values[i] - weekly_values[i - 1] for i in range(1, len(weekly_values))]

        weeks = list(range(2, len(weekly_values) + 1))
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.bar(weeks, improvements, color=colors, alpha=0.7)

        for bar, imp in zip(bars, improvements):
            height = float(bar.get_height())
            offset = max(0.01, abs(height) * 0.05)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (offset if height > 0 else -offset),
                f'{imp:+.3f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=8,
                clip_on=True,
            )

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title(f'Function {func_idx + 1} Week-to-Week Changes')
        ax.set_xlabel('Week')
        ax.set_ylabel('Improvement from Previous Week')
        ax.grid(True, alpha=0.3)

        total_improvement = float(sum(improvements))
        avg_improvement = float(np.mean(improvements)) if improvements else 0.0
        ax.text(
            0.02,
            0.98,
            f'Total: {total_improvement:+.3f}\nAvg: {avg_improvement:+.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=9,
        )

    return _finalize_figure(fig, 'Week-to-Week Improvements by Function')


def plot_overall_performance_summary(initial_data, weekly_outputs):
    """Plot overall performance summary across all functions."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    weekly_bests = []
    weekly_means = []
    for week_outputs in weekly_outputs:
        week_arr = np.asarray(week_outputs, dtype=float)
        weekly_bests.append(float(np.max(week_arr)))
        weekly_means.append(float(np.mean(week_arr)))

    initial_bests = [float(np.max(func.y)) for func in initial_data]
    initial_best_overall = float(np.max(initial_bests))
    initial_mean_overall = float(np.mean(initial_bests))
    weeks = list(range(1, len(weekly_bests) + 1))

    ax1.plot([0] + weeks, [initial_best_overall] + weekly_bests, 'o-', linewidth=2, markersize=8, label='Best Value')
    ax1.plot([0] + weeks, [initial_mean_overall] + weekly_means, 's-', linewidth=2, markersize=8, label='Mean Value')
    ax1.set_title('Overall Performance Trajectory')
    ax1.set_xlabel('Week (0 = Initial)')
    ax1.set_ylabel('Function Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    all_improvements = []
    for func_idx in range(8):
        weekly_values = [float(week_outputs[func_idx]) for week_outputs in weekly_outputs]
        best_weekly = float(np.max(weekly_values))
        initial_best = float(np.max(initial_data[func_idx].y))
        all_improvements.append(best_weekly - initial_best)

    colors = ['green' if imp > 0 else 'red' for imp in all_improvements]
    bars = ax2.bar(range(1, 9), all_improvements, color=colors, alpha=0.7)

    for bar, imp in zip(bars, all_improvements):
        height = float(bar.get_height())
        offset = max(0.03, abs(height) * 0.07)
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (offset if height > 0 else -offset),
            f'{imp:+.2f}',
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=9,
            clip_on=True,
        )

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Total Improvement by Function')
    ax2.set_xlabel('Function')
    ax2.set_ylabel('Improvement (Best Weekly - Initial Best)')
    ax2.set_xticks(range(1, 9))
    ax2.grid(True, alpha=0.3)

    successful_functions = sum(1 for imp in all_improvements if imp > 0)
    success_rate = successful_functions / 8 * 100
    stats_data = [successful_functions, 8 - successful_functions]
    labels = [f'Improved ({successful_functions})', f'No Improvement ({8 - successful_functions})']
    colors_pie = ['green', 'red']

    ax3.pie(stats_data, labels=labels, colors=colors_pie, autopct='%1.0f%%', startangle=90)
    ax3.set_title(f'Success Rate: {success_rate:.0f}%\nAvg Improvement: {np.mean(all_improvements):+.3f}')

    return _finalize_figure(fig, 'Overall BBO Performance Summary')


def plot_exploration_vs_exploitation_heatmap(weekly_outputs):
    """Create a heatmap showing the optimization landscape."""
    fig, ax = plt.subplots(figsize=(12, 8))

    output_matrix = np.array(weekly_outputs, dtype=float).T
    im = ax.imshow(output_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Function Value', rotation=270, labelpad=15)

    ax.set_xticks(range(len(weekly_outputs)))
    ax.set_xticklabels([f'Week {i + 1}' for i in range(len(weekly_outputs))])
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Function {i + 1}' for i in range(8)])

    for i in range(8):
        for j in range(len(weekly_outputs)):
            ax.text(j, i, f'{output_matrix[i, j]:.3f}', ha='center', va='center', color='black', fontsize=8)

    ax.set_title('Function Values Across Weeks (Heatmap)')
    ax.set_xlabel('Week')
    ax.set_ylabel('Function')
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.12, right=0.95)
    return fig


def save_all_plots():
    """Generate and save all visualization plots."""
    print("Loading data...")
    initial_data, weekly_inputs, weekly_outputs = load_data()

    output_dir = PROJECT_ROOT / "artifacts" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating function progress plot...")
    fig1 = plot_function_progress(initial_data, weekly_outputs)
    safe_savefig(fig1, output_dir / "function_progress.png", dpi=300)

    print("Generating week-to-week improvements plot...")
    fig2 = plot_week_to_week_improvements(weekly_outputs)
    safe_savefig(fig2, output_dir / "week_to_week_improvements.png", dpi=300)

    print("Generating overall performance summary...")
    fig3 = plot_overall_performance_summary(initial_data, weekly_outputs)
    safe_savefig(fig3, output_dir / "overall_performance_summary.png", dpi=300)

    print("Generating optimization heatmap...")
    fig4 = plot_exploration_vs_exploitation_heatmap(weekly_outputs)
    safe_savefig(fig4, output_dir / "optimization_heatmap.png", dpi=300)

    plt.show()
    plt.close('all')

    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated plots:")
    print("1. function_progress.png - Weekly progress for each function")
    print("2. week_to_week_improvements.png - Week-to-week changes")
    print("3. overall_performance_summary.png - Overall performance metrics")
    print("4. optimization_heatmap.png - Function values heatmap")


if __name__ == "__main__":
    save_all_plots()
