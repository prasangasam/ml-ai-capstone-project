from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from . import config


def decide_mode_maximise(y_last: float, y_hist: np.ndarray, tol_frac: float = config.EXPLOIT_TOL_FRAC_OF_RANGE) -> str:
    """Basic mode decision - maintained for compatibility."""
    y_hist = np.asarray(y_hist, float).reshape(-1)
    best = float(np.max(y_hist))
    worst = float(np.min(y_hist))
    span = max(best - worst, 1e-12)
    dist_frac = (best - float(y_last)) / span
    return "exploit" if dist_frac <= tol_frac else "explore"


def analyze_convergence(y_hist: np.ndarray, window: int = config.CONVERGENCE_WINDOW) -> Dict[str, float]:
    """Week 6: Advanced convergence analysis."""
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if len(y_hist) < window + 1:
        return {"convergence_rate": 0.0, "improvement_trend": 0.0, "stability_score": 0.0}

    recent_improvements = np.diff(y_hist[-window - 1:])
    convergence_rate = np.mean(np.abs(recent_improvements))
    improvement_trend = np.mean(recent_improvements)
    stability_score = 1.0 / (1.0 + np.std(recent_improvements))

    return {
        "convergence_rate": float(convergence_rate),
        "improvement_trend": float(improvement_trend),
        "stability_score": float(stability_score),
    }


def is_stagnating(y_hist: np.ndarray, window: int = 4, tol: float = 1e-3) -> bool:
    """Detect whether recent observations show little movement."""
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if len(y_hist) < window + 1:
        return False
    recent = y_hist[-window:]
    return bool((float(np.max(recent)) - float(np.min(recent))) < tol)


def choose_strategy(week: int, y_hist: np.ndarray) -> str:
    """Squirrel-inspired lightweight switching policy.

    - early rounds: broad exploration
    - middle rounds: model-based BO
    - late rounds: local refinement unless stagnating
    """
    if week <= 2:
        return "explore"
    if week <= 5:
        return "bo"
    if is_stagnating(y_hist):
        return "explore"
    return "refine"


def choose_acquisition(week: int, *, stagnating: bool = False, default: Optional[str] = None) -> str:
    """Small acquisition portfolio inspired by Squirrel's portfolio BO stage."""
    if stagnating:
        return "ucb"
    if default:
        base = default.lower().strip()
    else:
        cycle = ("ei", "pi", "ucb")
        base = cycle[(max(week, 1) - 1) % len(cycle)]
    return base


def adaptive_exploration_params(week: int, y_hist: np.ndarray, func_idx: int) -> Dict[str, float]:
    """Week 6: Sophisticated parameter tuning with adaptive exploration."""
    convergence_info = analyze_convergence(y_hist)

    # Adaptive exploration decay
    base_exploration = config.XI_EXPLORE * (config.ADAPTIVE_EXPLORATION_RATE ** week)

    # Multi-objective function portfolio balancing
    func_weight = config.MULTI_OBJECTIVE_WEIGHTS[func_idx] if func_idx < len(config.MULTI_OBJECTIVE_WEIGHTS) else 1.0

    # Enhanced uncertainty quantification
    uncertainty_factor = config.UNCERTAINTY_BOOST_FACTOR if convergence_info["stability_score"] < 0.5 else 1.0

    # Sophisticated parameter adaptation
    if convergence_info["improvement_trend"] < config.MIN_IMPROVEMENT_THRESHOLD:
        xi_adaptive = base_exploration * 1.5 * uncertainty_factor
        beta_adaptive = config.BETA_EXPLORE * func_weight * uncertainty_factor
        mode = "adaptive_explore"
    else:
        xi_adaptive = config.XI_EXPLOIT + (base_exploration * 0.3 * uncertainty_factor)
        beta_adaptive = config.BETA_EXPLOIT * func_weight
        mode = "adaptive_exploit"

    return {
        "xi": float(xi_adaptive),
        "beta": float(beta_adaptive),
        "mode": mode,
        "convergence_rate": convergence_info["convergence_rate"],
        "improvement_trend": convergence_info["improvement_trend"],
        "stability_score": convergence_info["stability_score"],
        "func_weight": float(func_weight),
        "uncertainty_factor": float(uncertainty_factor),
    }


def tune_params(mode: str, acquisition: str, week: Optional[int] = None, y_hist: Optional[np.ndarray] = None, func_idx: Optional[int] = None) -> Dict[str, float]:
    """Enhanced parameter tuning with Week 6 advanced features."""
    a = acquisition.lower().strip()

    if week is not None and y_hist is not None and func_idx is not None and week >= 6:
        return adaptive_exploration_params(week, y_hist, func_idx)

    if a in ("ei", "pi"):
        return {"xi": config.XI_EXPLOIT if mode == "exploit" else config.XI_EXPLORE}
    if a == "ucb":
        return {"beta": config.BETA_EXPLOIT if mode == "exploit" else config.BETA_EXPLORE}
    raise ValueError("ACQUISITION must be one of: ei, pi, ucb")


def multi_objective_portfolio_balance(all_func_performances: List[float]) -> Dict[int, float]:
    """Week 6: Multi-objective balancing across function portfolios."""
    performances = np.array(all_func_performances, dtype=float)
    if performances.size == 0:
        return {}
    span = float(performances.max() - performances.min())
    if span <= 1e-12:
        return {i: 1.0 for i in range(len(performances))}

    normalized_perf = (performances - performances.min()) / span

    # Assign higher weights to underperforming functions
    inverse_weights = 1.0 - normalized_perf
    balanced_weights = inverse_weights / inverse_weights.sum() * len(performances)

    return {i: float(weight) for i, weight in enumerate(balanced_weights)}
