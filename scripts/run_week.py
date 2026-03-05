from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add the src directory to Python path so we can import bbo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bbo.pipeline import run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial_dir", type=str, required=True)
    ap.add_argument("--weekly_dir", type=str, required=True)
    args = ap.parse_args()

    result = run(initial_dir=Path(args.initial_dir), weekly_dir=Path(args.weekly_dir))
    print(f"\nLoaded weekly data mode: {result['weekly_mode']}")
    print("\n=== NEXT WEEK QUERIES (PORTAL FORMAT) ===")
    for line in result["portal_lines"]:
        print(line)
    print("\nSaved:", result["submission_path"])
    print("Snapshot:", result["snapshot_path"])

if __name__ == "__main__":
    main()
