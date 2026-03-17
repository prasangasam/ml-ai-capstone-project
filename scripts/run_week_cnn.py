#!/usr/bin/env python3
"""Enhanced run_week.py with CNN integration for BBO optimization"""

from pathlib import Path
import argparse
import sys
import time
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bbo.data_loader import load_initial_from_dir, load_weekly
from bbo import io
from bbo import config


def should_use_cnn(dim, n_points, week_k=None):
    """Determine if CNN should be used based on function characteristics"""
    # Handle week_k being a string like "matrix"
    week_num = 6 if isinstance(week_k, str) else week_k
    
    # CNN usage criteria from integration guide
    if dim >= 4 and n_points >= 25:
        return True, f"4D+ function with {n_points} points - CNN beneficial"
    elif week_num and week_num >= 4 and dim >= 3 and n_points >= 20:
        return True, f"Week {week_num}: 3D+ function with sufficient data"
    else:
        return False, f"Using GP: {dim}D function with {n_points} points"


def run_hybrid_optimization(initial_dir, weekly_dir, use_cnn=False, cnn_weight=0.3, week_k=None):
    """Run optimization with optional CNN enhancement"""
    
    # Load data
    print(f"🔄 Loading initial data from {initial_dir}")
    funcs = load_initial_from_dir(initial_dir)
    
    print(f"🔄 Loading weekly data from {weekly_dir}")
    weekly_inputs, weekly_outputs, week_k_loaded = load_weekly(weekly_dir)
    
    if week_k is None:
        week_k = week_k_loaded
    
    # Handle week_k being a string like "matrix" - assume week 6 for advanced features
    week_num = 6 if isinstance(week_k, str) else week_k
    
    print(f"📊 Running optimization for Week {week_num}")
    print(f"🎯 Found {len(funcs)} functions")
    
    # Prepare for optimization
    xi_params = []
    results = []
    optimization_reports = []
    
    # Import optimization functions
    if use_cnn:
        print("🤖 CNN integration enabled")
        try:
            from bbo.cnn_surrogate import propose_next_point_cnn
            from bbo.gp import propose_next_point
            cnn_available = True
        except ImportError as e:
            print(f"⚠️ CNN import failed: {e}")
            print("🔄 Falling back to GP-only optimization")
            from bbo.gp import propose_next_point
            cnn_available = False
            use_cnn = False
    else:
        print("📊 Standard GP optimization")
        from bbo.gp import propose_next_point
        cnn_available = False
    
    # Week 6 advanced parameters
    try:
        from bbo.strategy import adaptive_exploration_params
        week6_available = True
        print("✨ Week 6 advanced features available")
    except (ImportError, AttributeError):
        week6_available = False
        print("⚪ Using standard parameters")
    
    # Process each function
    for i, f in enumerate(funcs):
        func_name = f"Function {i+1}"
        print(f"\n🎯 Processing {func_name}")
        
        # Add weekly evaluations to function data
        current_X = f.X.copy()
        current_y = f.y.copy()
        
        for week_in, week_out in zip(weekly_inputs, weekly_outputs):
            if i < len(week_in) and i < len(week_out):
                current_X = np.vstack([current_X, week_in[i].reshape(1, -1)])
                current_y = np.append(current_y, week_out[i])
        
        n_points = len(current_y)
        dim = current_X.shape[1]
        print(f"   📈 {n_points} evaluations in {dim}D space")
        
        # Determine optimization method
        use_cnn_for_func = False
        method_reason = ""
        
        if use_cnn and cnn_available:
            use_cnn_for_func, method_reason = should_use_cnn(dim, n_points, week_k)
        else:
            method_reason = "GP (standard)"
        
        print(f"   🔧 Method: {method_reason}")
        
        # Get optimization parameters
        if week6_available and week_num >= 6:
            # Use Week 6 advanced parameters
            params = adaptive_exploration_params(week_num, current_y, i)
            xi = params["xi"]
            beta = params["beta"]
            print(f"   ✨ Advanced params: xi={xi:.3f}, beta={beta:.3f}")
        else:
            # Standard parameters
            xi = config.EXPLORATION_PARAMETER
            beta = config.CONFIDENCE_PARAMETER
            print(f"   📊 Standard params: xi={xi:.3f}, beta={beta:.3f}")
        
        xi_params.append(xi)
        
        # Run optimization
        start_time = time.time()
        
        # Use memory-safe parameters
        safe_n_candidates = min(config.N_CANDIDATES, 5000)  # Cap at 5000 for memory safety
        
        if use_cnn_for_func:
            try:
                # Use CNN optimization with faster training for testing
                x_next, report = propose_next_point_cnn(
                    current_X, current_y,
                    acquisition=config.ACQUISITION,
                    xi=xi,
                    seed=config.RNG_SEED + 31*i,
                    n_candidates=safe_n_candidates,
                    max_epochs=50,  # Reduced epochs for faster processing
                )
                report["method_used"] = "cnn_surrogate"
                print(f"   🤖 CNN optimization completed")
                
            except Exception as e:
                print(f"   ⚠️ CNN failed: {e}")
                print(f"   🔄 Falling back to GP...")
                
                # Fallback to GP
                x_next, report = propose_next_point(
                    current_X, current_y,
                    acquisition=config.ACQUISITION,
                    xi=xi, beta=beta,
                    seed=config.RNG_SEED + 31*i,
                    n_candidates=safe_n_candidates,
                )
                report["method_used"] = "gp_fallback"
                report["cnn_error"] = str(e)
        else:
            # Use standard GP optimization
            x_next, report = propose_next_point(
                current_X, current_y,
                acquisition=config.ACQUISITION,
                xi=xi, beta=beta,
                seed=config.RNG_SEED + 31*i,
                n_candidates=safe_n_candidates,
            )
            report["method_used"] = "gp"
        
        opt_time = time.time() - start_time
        print(f"   ⏱️ Optimization time: {opt_time:.2f}s")
        print(f"   📍 Next point: {x_next}")
        
        # Store results as portal lines (formatted for submission)
        portal_line = io.fmt_query(x_next)
        results.append(portal_line)
        
        # Enhanced report
        report["function_idx"] = i
        report["function_dim"] = dim
        report["n_evaluations"] = n_points
        report["optimization_time"] = opt_time
        report["method_reason"] = method_reason
        optimization_reports.append(report)
    
    # Save results
    submission_path = io.save_submission_file(week_next=week_num+1, portal_lines=results)
    print(f"\n💾 Results saved to: {submission_path}")
    
    # Print summary
    print(f"\n📋 Optimization Summary:")
    print(f"   Week: {week_num}")
    print(f"   Functions optimized: {len(results)}")
    
    if use_cnn:
        cnn_count = sum(1 for r in optimization_reports if r["method_used"] == "cnn_surrogate")
        gp_count = len(optimization_reports) - cnn_count
        print(f"   CNN optimizations: {cnn_count}")
        print(f"   GP optimizations: {gp_count}")
        
        # Show which functions used CNN
        cnn_functions = [r["function_idx"]+1 for r in optimization_reports if r["method_used"] == "cnn_surrogate"]
        if cnn_functions:
            print(f"   CNN used for functions: {cnn_functions}")
    
    return {
        'week_k': week_num,
        'submission_path': submission_path,
        'optimization_reports': optimization_reports,
        'portal_lines': results,
        'xi_params': xi_params
    }


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced BBO optimization with CNN integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--initial_dir', type=Path, required=True,
                       help='Directory containing initial function data')
    parser.add_argument('--weekly_dir', type=Path, required=True,
                       help='Directory containing weekly evaluation data')
    
    # CNN options
    parser.add_argument('--use_cnn', action='store_true',
                       help='Enable CNN-enhanced optimization')
    parser.add_argument('--cnn_weight', type=float, default=0.3,
                       help='Weight for CNN in ensemble (0.0-1.0)')
    
    # Advanced options
    parser.add_argument('--week', type=int,
                       help='Override week number (default: auto-detect)')
    parser.add_argument('--force_cnn', action='store_true',
                       help='Force CNN usage even for low-dimensional functions')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Enhanced BBO Optimization with CNN Integration")
    print("=" * 60)
    
    # Run optimization
    result = run_hybrid_optimization(
        initial_dir=args.initial_dir,
        weekly_dir=args.weekly_dir,
        use_cnn=args.use_cnn,
        cnn_weight=args.cnn_weight,
        week_k=args.week
    )
    
    print(f"\n✅ Optimization completed successfully!")
    print(f"📄 Check results in: {result['submission_path']}")


if __name__ == "__main__":
    main()