#!/usr/bin/env python3
"""Simple CNN integration example - modify your existing run_week.py"""

from pathlib import Path
import sys
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import the enhanced pipeline with CNN support
from bbo.pipeline import run

def main():
    """Simple CNN-enabled optimization"""
    parser = argparse.ArgumentParser(description="BBO optimization with optional CNN")
    parser.add_argument('--initial_dir', type=Path, required=True)
    parser.add_argument('--weekly_dir', type=Path, required=True)
    
    # CNN options
    parser.add_argument('--use_cnn', action='store_true', 
                       help='Enable CNN for high-dimensional functions')
    parser.add_argument('--force_cnn', action='store_true',
                       help='Force CNN even for low-dimensional functions')
    parser.add_argument('--cnn_functions', type=int, nargs='+',
                       help='Specific functions to use CNN (1-8)')
    
    args = parser.parse_args()
    
    print(f"🚀 Running BBO optimization...")
    print(f"   Initial data: {args.initial_dir}")
    print(f"   Weekly data: {args.weekly_dir}")
    
    if args.use_cnn:
        print(f"   🤖 CNN integration: ENABLED")
        if args.force_cnn:
            print(f"   ⚡ Force CNN: YES")
        if args.cnn_functions:
            print(f"   🎯 CNN functions: {args.cnn_functions}")
    else:
        print(f"   📊 Using standard GP optimization")
    
    try:
        # Run optimization with CNN support
        result = run(
            initial_dir=args.initial_dir,
            weekly_dir=args.weekly_dir,
            use_cnn=args.use_cnn,
            force_cnn=args.force_cnn,
            cnn_functions=args.cnn_functions
        )
        
        print(f"\n✅ Optimization completed!")
        print(f"📄 Week {result['week_k']} results: {result['submission_path']}")
        
    except ImportError as e:
        if "torch" in str(e):
            print(f"\n⚠️ CNN features require PyTorch")
            print(f"🔧 Install with: pip install torch torchvision")
            print(f"🔄 Falling back to standard optimization...")
            
            # Run without CNN
            result = run(
                initial_dir=args.initial_dir,
                weekly_dir=args.weekly_dir
            )
            print(f"✅ Standard optimization completed!")
        else:
            raise
    
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())