#!/usr/bin/env python3
"""Demo script showing CNN integration with actual BBO function data"""

from pathlib import Path
import sys
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bbo.data_loader import load_initial_from_dir, load_weekly
from bbo.pipeline import run
from bbo import config

def analyze_function_characteristics():
    """Analyze characteristics of each function to determine CNN suitability"""
    print("🔍 Analyzing Function Characteristics for CNN Suitability")
    print("=" * 60)
    
    # Load data
    initial_dir = Path("data/initial_data")
    weekly_dir = Path("data/weekly")
    
    funcs = load_initial_from_dir(initial_dir)
    weekly_inputs, weekly_outputs, week_k = load_weekly(weekly_dir)
    
    print(f"📊 Current dataset: Week {week_k}")
    print(f"🎯 Functions analyzed: {len(funcs)}")
    
    cnn_recommendations = []
    
    for i, f in enumerate(funcs, start=1):
        # Calculate total evaluations (initial + weekly)
        current_X = f.X.copy()
        current_y = f.y.copy()
        
        for week_in, week_out in zip(weekly_inputs, weekly_outputs):
            if i-1 < len(week_in) and i-1 < len(week_out):
                current_X = np.vstack([current_X, week_in[i-1].reshape(1, -1)])
                current_y = np.append(current_y, week_out[i-1])
        
        n_points = len(current_y)
        dim = current_X.shape[1]
        
        # Determine recommendation
        # Handle week_k being a string like "matrix"
        week_num = 6 if isinstance(week_k, str) else week_k
        
        if dim >= 4 and n_points >= 25:
            recommendation = "✅ CNN Recommended"
            reason = f"{dim}D function with {n_points} evaluations - high dimensional"
            use_cnn = True
        elif dim >= 3 and n_points >= 20 and week_num >= 4:
            recommendation = "🟡 CNN Possible"
            reason = f"{dim}D function with {n_points} evaluations - moderate case"
            use_cnn = True
        else:
            recommendation = "⚪ GP Preferred"
            reason = f"{dim}D function with {n_points} evaluations - GP sufficient"
            use_cnn = False
        
        print(f"\n📈 Function {i}:")
        print(f"   Dimensions: {dim}D")
        print(f"   Evaluations: {n_points}")
        print(f"   Recommendation: {recommendation}")
        print(f"   Reason: {reason}")
        
        if use_cnn:
            cnn_recommendations.append(i)
    
    return cnn_recommendations


def demo_cnn_optimization():
    """Demonstrate CNN optimization with actual function data"""
    print("\n" + "=" * 60)
    print("🤖 Demo: CNN-Enhanced BBO Optimization")
    print("=" * 60)
    
    # Analyze functions first
    cnn_functions = analyze_function_characteristics()
    
    if not cnn_functions:
        print("\n⚠️ No functions suitable for CNN optimization based on current data")
        print("   Using standard GP optimization for demo")
    else:
        print(f"\n🎯 Recommended CNN functions: {cnn_functions}")
    
    print(f"\n🚀 Running optimization with CNN integration...")
    
    try:
        # Run CNN-enhanced optimization
        result = run(
            initial_dir=Path("data/initial_data"),
            weekly_dir=Path("data/weekly"),
            use_cnn=True,
            force_cnn=False,
            cnn_functions=cnn_functions
        )
        
        print(f"\n✅ Optimization completed successfully!")
        print(f"📄 Week {result['week_k']} results saved to: {result['submission_path']}")
        
        # Analyze which methods were used
        if 'diagnostics' in result:
            print(f"\n📊 Method Usage Summary:")
            
            method_counts = {}
            for diag in result.get('diagnostics', []):
                method = diag.get('method_used', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
            
            for method, count in method_counts.items():
                print(f"   {method}: {count} functions")
        
    except ImportError as e:
        print(f"\n⚠️ CNN optimization failed due to missing dependencies:")
        print(f"   {e}")
        print(f"🔧 To enable CNN features, install: pip install torch torchvision")
        print(f"\n🔄 Running standard GP optimization instead...")
        
        # Fallback to standard optimization
        result = run(
            initial_dir=Path("data/initial_data"),
            weekly_dir=Path("data/weekly")
        )
        
        print(f"✅ Standard optimization completed!")
        print(f"📄 Results saved to: {result['submission_path']}")
    
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


def demo_individual_cnn_usage():
    """Demonstrate individual CNN function usage"""
    print("\n" + "=" * 60)
    print("🔬 Demo: Individual CNN Function Usage")
    print("=" * 60)
    
    try:
        from bbo.cnn_surrogate import propose_next_point_cnn, CNNBayesianOptimizer
        from bbo.gp import propose_next_point
        
        # Load actual function data
        funcs = load_initial_from_dir(Path("data/initial_data"))
        weekly_inputs, weekly_outputs, week_k = load_weekly(Path("data/weekly"))
        
        # Test on Function 4 (4D - good candidate for CNN)
        func_idx = 3  # Function 4 (0-indexed)
        if func_idx < len(funcs):
            f = funcs[func_idx]
            
            # Add weekly data
            current_X = f.X.copy()
            current_y = f.y.copy()
            
            for week_in, week_out in zip(weekly_inputs, weekly_outputs):
                if func_idx < len(week_in) and func_idx < len(week_out):
                    current_X = np.vstack([current_X, week_in[func_idx].reshape(1, -1)])
                    current_y = np.append(current_y, week_out[func_idx])
            
            print(f"🎯 Testing on Function {func_idx+1}")
            print(f"   Dimensions: {current_X.shape[1]}D")
            print(f"   Evaluations: {len(current_y)}")
            
            # Compare GP vs CNN
            print(f"\n🔄 Comparing optimization methods...")
            
            # Test GP
            print(f"   📊 Testing GP...")
            x_gp, report_gp = propose_next_point(
                current_X, current_y,
                acquisition="ei",
                xi=0.01, beta=1.0,
                seed=42,
                n_candidates=1000
            )
            
            # Test CNN
            print(f"   🤖 Testing CNN...")
            x_cnn, report_cnn = propose_next_point_cnn(
                current_X, current_y,
                acquisition="ei",
                xi=0.01,
                seed=42,
                n_candidates=1000
            )
            
            print(f"\n📈 Results:")
            print(f"   GP next point: {x_gp}")
            print(f"   GP EI score: {report_gp.get('ei_score', 'N/A')}")
            print(f"   CNN next point: {x_cnn}")
            print(f"   CNN EI score: {report_cnn.get('ei_score', 'N/A')}")
            
            # Check point difference
            point_diff = np.linalg.norm(x_gp - x_cnn)
            print(f"   Point difference (L2): {point_diff:.4f}")
            
            if point_diff < 0.1:
                print(f"   ✅ Methods agree closely")
            else:
                print(f"   🔍 Methods propose different regions")
        
    except ImportError as e:
        print(f"⚠️ CNN libraries not available: {e}")
        print(f"🔧 Install with: pip install torch torchvision")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function"""
    print("🚀 BBO CNN Integration Demo")
    print("=" * 60)
    
    # Check if we can access the data
    initial_dir = Path("data/initial_data")
    weekly_dir = Path("data/weekly")
    
    if not initial_dir.exists():
        print(f"❌ Initial data directory not found: {initial_dir}")
        print(f"   Please run this script from the project root directory")
        return
    
    if not weekly_dir.exists():
        print(f"❌ Weekly data directory not found: {weekly_dir}")
        print(f"   Please run this script from the project root directory")
        return
    
    # Run demonstrations
    try:
        # 1. Analyze function characteristics
        analyze_function_characteristics()
        
        # 2. Demo full CNN-enhanced optimization
        demo_cnn_optimization()
        
        # 3. Demo individual CNN usage
        demo_individual_cnn_usage()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print(f"✅ CNN Integration Demo Completed")
    print(f"=" * 60)


if __name__ == "__main__":
    main()