from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config, io
from .data_loader import load_initial_from_dir, load_weekly
from .gp import propose_next_point
from .strategy import (
    choose_acquisition,
    choose_strategy,
    decide_mode_maximise,
    llm_strategy_metadata,
    multi_objective_portfolio_balance,
    tune_params,
    is_stagnating,
    recent_instability,
)

# CNN integration (optional)
try:
    from .cnn_surrogate import propose_next_point_cnn
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False


@dataclass
class FunctionDataset:
    idx: int
    X: np.ndarray
    y: np.ndarray

    def append(self, x_new: np.ndarray, y_new: float) -> None:
        self.X = np.vstack([self.X, np.asarray(x_new, float).reshape(1, -1)])
        self.y = np.concatenate([self.y, [float(y_new)]])


def should_use_cnn(dim: int, n_points: int, week_k: Optional[int] = None, force_cnn: bool = False):
    """Determine if CNN should be used for optimization."""
    if not CNN_AVAILABLE:
        return False, "CNN not available"

    if force_cnn:
        return True, "CNN forced by user"

    if dim >= 4 and n_points >= 25:
        return True, f"{dim}D function with {n_points} points"
    if week_k and week_k >= 4 and dim >= 3 and n_points >= 20:
        return True, f"Week {week_k}: sufficient data for CNN"
    return False, f"GP preferred: {dim}D with {n_points} points"


def _safe_portfolio_weights(last_week_outputs: np.ndarray) -> Dict[int, float]:
    values = np.asarray(last_week_outputs, dtype=float).reshape(-1)
    if len(values) == 0:
        return {}
    if np.allclose(values.max(), values.min()):
        return {i: 1.0 for i in range(len(values))}
    return multi_objective_portfolio_balance(values.tolist())


def _extract_acq_params(tuned: Dict[str, float], portfolio_weight: float) -> Dict[str, float]:
    xi = float(tuned.get("xi", config.XI_EXPLORE))
    beta = float(tuned.get("beta", config.BETA_EXPLORE)) * float(portfolio_weight)
    return {"xi": xi, "beta": beta}


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
    if week_k == 0:
        raise ValueError("No weekly outputs found; cannot generate next-week queries.")

    last_week_outputs = np.asarray(weekly_outputs_all[-1], float).reshape(-1)
    portfolio_weights = _safe_portfolio_weights(last_week_outputs)

    portal_lines: List[str] = []
    diagnostics: List[Dict[str, Any]] = []

    for i, f in enumerate(funcs, start=1):
        func_idx = i - 1
        y_last = float(last_week_outputs[func_idx])
        base_mode = decide_mode_maximise(y_last, f.y)
        stagnating = is_stagnating(f.y)
        instability = recent_instability(f.y)
        strategy = choose_strategy(week_k, f.y)
        acquisition_name = choose_acquisition(
            week_k,
            stagnating=stagnating,
            instability=instability,
            strategy=strategy,
            default=config.ACQUISITION,
        )

        tuned = tune_params(
            base_mode,
            acquisition_name,
            week=week_k,
            y_hist=f.y,
            func_idx=func_idx,
        )

        portfolio_weight = float(portfolio_weights.get(func_idx, 1.0))
        acq_params = _extract_acq_params(tuned, portfolio_weight)
        llm_meta = llm_strategy_metadata(
            dim=f.X.shape[1],
            strategy=strategy,
            instability=instability,
            n_observations=len(f.y),
        )

        use_cnn_for_func = False
        method_reason = "GP (standard)"
        if use_cnn:
            use_cnn_for_func, method_reason = should_use_cnn(
                f.X.shape[1], len(f.y), week_k, force_cnn
            )
            if cnn_functions and i in cnn_functions:
                use_cnn_for_func = True
                method_reason = f"CNN forced for function {i}"

        if use_cnn_for_func:
            try:
                x_next, report = propose_next_point_cnn(
                    f.X,
                    f.y,
                    acquisition=acquisition_name,
                    xi=acq_params["xi"],
                    seed=config.RNG_SEED + 31 * i,
                    n_candidates=config.N_CANDIDATES,
                )
                report["method_used"] = "cnn_surrogate"
                report["method_reason"] = method_reason
                report["strategy"] = strategy
                report["acquisition_used"] = acquisition_name
                report["instability"] = float(instability)
            except Exception as e:
                x_next, report = propose_next_point(
                    f.X,
                    f.y,
                    acquisition=acquisition_name,
                    xi=acq_params["xi"],
                    beta=acq_params["beta"],
                    seed=config.RNG_SEED + 31 * i,
                    n_candidates=config.N_CANDIDATES,
                    strategy=strategy,
                    instability=instability,
                )
                report["method_used"] = "gp_fallback"
                report["cnn_error"] = str(e)
                report["method_reason"] = f"CNN failed: {method_reason}"
        else:
            x_next, report = propose_next_point(
                f.X,
                f.y,
                acquisition=acquisition_name,
                xi=acq_params["xi"],
                beta=acq_params["beta"],
                seed=config.RNG_SEED + 31 * i,
                n_candidates=config.N_CANDIDATES,
                strategy=strategy,
                instability=instability,
            )
            report["method_used"] = "gp"
            report["method_reason"] = method_reason

        portal_lines.append(io.fmt_query(x_next))
        diagnostics.append({
            "function_index": i,
            "base_mode": base_mode,
            "effective_mode": tuned.get("mode", base_mode),
            "strategy": strategy,
            "stagnating": stagnating,
            "instability": float(instability),
            "portfolio_weight": portfolio_weight,
            "xi": acq_params["xi"],
            "beta": acq_params["beta"],
            "acquisition": acquisition_name,
            "y_last": y_last,
            "y_best_so_far": float(np.max(f.y)),
            "n_observations": int(len(f.y)),
            **llm_meta,
            **tuned,
            **report,
        })

    submission_path = io.save_submission_file(week_next=week_k + 1, portal_lines=portal_lines)
    snapshot_path = io.save_week_snapshot(week_k=week_k, payload={
        "acquisition": config.ACQUISITION,
        "week_k_observed": week_k,
        "initial_dir": str(initial_dir),
        "weekly_dir": str(weekly_dir),
        "weekly_mode": weekly_mode,
        "portfolio_weights": portfolio_weights,
        "next_week_portal_lines": portal_lines,
        "diagnostics": diagnostics,
    })

    return {
        "week_k": week_k,
        "portal_lines": portal_lines,
        "submission_path": str(submission_path),
        "snapshot_path": str(snapshot_path),
        "weekly_mode": weekly_mode,
        "portfolio_weights": portfolio_weights,
    }
