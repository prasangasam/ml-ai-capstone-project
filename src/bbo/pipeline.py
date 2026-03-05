from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from . import config
from .data_loader import load_initial_from_dir, load_weekly
from .strategy import decide_mode_maximise, tune_params
from .gp import propose_next_point
from . import io

# CNN integration (optional)
try:
    from .cnn_surrogate import propose_next_point_cnn
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

# Week 6 advanced features (optional)
try:
    from .strategy import adaptive_exploration_params
    WEEK6_AVAILABLE = True
except (ImportError, AttributeError):
    WEEK6_AVAILABLE = False

@dataclass
class FunctionDataset:
    idx: int
    X: np.ndarray
    y: np.ndarray

    def append(self, x_new: np.ndarray, y_new: float) -> None:
        self.X = np.vstack([self.X, np.asarray(x_new, float).reshape(1, -1)])
        self.y = np.concatenate([self.y, [float(y_new)]])

def should_use_cnn(dim, n_points, week_k=None, force_cnn=False):
    """Determine if CNN should be used for optimization"""
    if not CNN_AVAILABLE:
        return False, "CNN not available"
    
    if force_cnn:
        return True, "CNN forced by user"
    
    if dim >= 4 and n_points >= 25:
        return True, f"4D+ function with {n_points} points"
    elif week_k and week_k >= 4 and dim >= 3 and n_points >= 20:
        return True, f"Week {week_k}: sufficient data for CNN"
    else:
        return False, f"GP preferred: {dim}D with {n_points} points"

def run(*, initial_dir: Path, weekly_dir: Path, use_cnn: bool = False, force_cnn: bool = False, cnn_functions: List[int] = None) -> Dict[str, Any]:
    seeds = load_initial_from_dir(initial_dir)
    weekly_inputs_all, weekly_outputs_all, weekly_mode = load_weekly(weekly_dir)

    funcs: List[FunctionDataset] = [
        FunctionDataset(s.idx, np.asarray(s.X, float), np.asarray(s.y, float).reshape(-1))
        for s in seeds
    ]

    for week_inputs, week_outputs in zip(weekly_inputs_all, weekly_outputs_all):
        for i in range(8):
            funcs[i].append(week_inputs[i], week_outputs[i])

    week_k = len(weekly_outputs_all)
    last_week_outputs = np.asarray(weekly_outputs_all[-1], float).reshape(-1)

    portal_lines: List[str] = []
    diagnostics: List[Dict[str, Any]] = []

    for i, f in enumerate(funcs, start=1):
        mode = decide_mode_maximise(float(last_week_outputs[i-1]), f.y)
        tuned = tune_params(mode, config.ACQUISITION)
        xi = float(tuned.get("xi", config.XI_EXPLORE))
        beta = float(tuned.get("beta", config.BETA_EXPLORE))

        # Determine if we should use CNN for this function
        use_cnn_for_func = False
        method_reason = "GP (standard)"
        
        if use_cnn:
            # Check CNN criteria
            use_cnn_for_func, method_reason = should_use_cnn(
                f.X.shape[1], len(f.y), week_k, force_cnn
            )
            
            # Override for specific functions if requested
            if cnn_functions and i in cnn_functions:
                use_cnn_for_func = True
                method_reason = f"CNN forced for function {i}"

        # Propose next point for evaluation
        if use_cnn_for_func:
            try:
                # Use CNN optimization
                x_next, report = propose_next_point_cnn(
                    f.X, f.y,
                    acquisition=config.ACQUISITION,
                    xi=xi,
                    seed=config.RNG_SEED + 31*i,
                    n_candidates=config.N_CANDIDATES,
                )
                report["method_used"] = "cnn_surrogate"
                report["method_reason"] = method_reason
                
            except Exception as e:
                # Fallback to GP if CNN fails
                print(f"CNN failed for function {i}: {e}. Using GP fallback.")
                x_next, report = propose_next_point(
                    f.X, f.y,
                    acquisition=config.ACQUISITION,
                    xi=xi, beta=beta,
                    seed=config.RNG_SEED + 31*i,
                    n_candidates=config.N_CANDIDATES,
                )
                report["method_used"] = "gp_fallback"
                report["cnn_error"] = str(e)
                report["method_reason"] = f"CNN failed: {method_reason}"
        else:
            # Use standard GP optimization
            x_next, report = propose_next_point(
                f.X, f.y,
                acquisition=config.ACQUISITION,
                xi=xi, beta=beta,
                seed=config.RNG_SEED + 31*i,
                n_candidates=config.N_CANDIDATES,
            )
            report["method_used"] = "gp"
            report["method_reason"] = method_reason

        portal_lines.append(io.fmt_query(x_next))
        diagnostics.append({"function_index": i, "mode": mode, "xi": xi, "beta": beta, **report})

    submission_path = io.save_submission_file(week_next=week_k+1, portal_lines=portal_lines)
    snapshot_path = io.save_week_snapshot(week_k=week_k, payload={
        "acquisition": config.ACQUISITION,
        "week_k_observed": week_k,
        "initial_dir": str(initial_dir),
        "weekly_dir": str(weekly_dir),
        "weekly_mode": weekly_mode,
        "next_week_portal_lines": portal_lines,
        "diagnostics": diagnostics,
    })

    return {
        "week_k": week_k,
        "portal_lines": portal_lines,
        "submission_path": str(submission_path),
        "snapshot_path": str(snapshot_path),
        "weekly_mode": weekly_mode,
    }
