#!/usr/bin/env python3
"""
Week 6 Advanced Optimization Demo

This script demonstrates the sophisticated parameter tuning, convergence analysis, 
and multi-objective balancing features that would be used in Week 6 optimization.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bbo.data_loader import load_initial_from_dir, load_weekly
from bbo.strategy import (
    adaptive_exploration_params, 
    analyze_convergence, 
    multi_objective_portfolio_balance,
    tune_params
)
from bbo import config, io

def demo_week6_features():
    """Demonstrate Week 6 advanced optimization features"""
    
    print("🚀 Week 6 Advanced Optimization Demo")
    print("=" * 50)
    
    # Load actual data
    initial_dir = Path("data/initial_data")
    weekly_dir = Path("data/weekly") 
    
    seeds = load_initial_from_dir(initial_dir)
    weekly_inputs_all, weekly_outputs_all, weekly_mode = load_weekly(weekly_dir)
    
    # Simulate Week 6 by extending data
    week_k = len(weekly_outputs_all)
    simulated_week = 6
    last_week_outputs = np.asarray(weekly_outputs_all[-1], float).reshape(-1)
    
    print(f"📊 Current week: {week_k}, simulating Week {simulated_week}")
    print(f"📈 Last week outputs: {last_week_outputs}")
    print()
    
    # Multi-objective portfolio balancing
    print("🎯 Multi-Objective Portfolio Balancing:")
    portfolio_weights = multi_objective_portfolio_balance(last_week_outputs.tolist())
    for i, weight in portfolio_weights.items():
        print(f"   Function {i+1}: weight = {weight:.3f}")
    print()
    
    # Enhanced optimization for each function
    print("🧠 Advanced Parameter Tuning & Convergence Analysis:")
    print("-" * 60)
    
    enhanced_diagnostics = []
    
    for i, seed in enumerate(seeds):
        func_idx = i
        func_num = i + 1
        
        # Build historical performance for this function
        y_hist = list(seed.y)
        for week_outputs in weekly_outputs_all:
            y_hist.append(week_outputs[i])
        y_hist = np.array(y_hist)
        
        print(f"\n📍 Function {func_num} (Dimensionality: {seed.X.shape[1]}D):")
        
        # Convergence analysis
        conv_analysis = analyze_convergence(y_hist)
        print(f"   🔄 Convergence rate: {conv_analysis['convergence_rate']:.6f}")
        print(f"   📈 Improvement trend: {conv_analysis['improvement_trend']:.6f}")
        print(f"   ⚖️ Stability score: {conv_analysis['stability_score']:.3f}")
        
        # Week 6 advanced parameters
        if simulated_week >= 6:
            advanced_params = adaptive_exploration_params(simulated_week, y_hist, func_idx)
            print(f"   🎛️ Advanced mode: {advanced_params['mode']}")
            print(f"   🔍 Adaptive xi: {advanced_params['xi']:.6f}")
            print(f"   🔍 Adaptive beta: {advanced_params['beta']:.3f}")
            print(f"   ⚡ Uncertainty factor: {advanced_params['uncertainty_factor']:.1f}")
            print(f"   🎪 Portfolio weight: {portfolio_weights[func_idx]:.3f}")
        else:
            # Legacy parameters for comparison
            legacy_params = tune_params("explore", config.ACQUISITION)
            print(f"   📊 Legacy xi: {legacy_params.get('xi', 'N/A')}")
            print(f"   📊 Legacy beta: {legacy_params.get('beta', 'N/A')}")
            advanced_params = legacy_params
        
        enhanced_diagnostics.append({
            'function': func_num,
            'dimensionality': seed.X.shape[1],
            'historical_performance': y_hist,
            'convergence_analysis': conv_analysis,
            'advanced_parameters': advanced_params,
            'portfolio_weight': portfolio_weights.get(func_idx, 1.0)
        })
    
    print("\n" + "=" * 60)
    print("📊 Week 6 Advanced Optimization Summary")
    print("=" * 60)
    
    # Summary statistics
    stability_scores = [d['convergence_analysis']['stability_score'] for d in enhanced_diagnostics]
    improvement_trends = [d['convergence_analysis']['improvement_trend'] for d in enhanced_diagnostics]
    
    print(f"🎯 Portfolio Balancing: {len([w for w in portfolio_weights.values() if w > 1.2])} functions prioritized")
    print(f"⚖️ Average Stability: {np.mean(stability_scores):.3f}")
    print(f"📈 Average Improvement: {np.mean(improvement_trends):.6f}")
    print(f"🔧 Sophisticated Parameter Tuning: ACTIVE")
    print(f"🧮 Enhanced Uncertainty Quantification: ACTIVE")
    print(f"🎪 Multi-Objective Balancing: ACTIVE")
    print(f"📊 Convergence Analysis: {len(enhanced_diagnostics)} functions monitored")
    
    print("\n✅ Week 6 Advanced Optimization Features Demonstrated!")
    
    return enhanced_diagnostics

if __name__ == "__main__":
    demo_week6_features()