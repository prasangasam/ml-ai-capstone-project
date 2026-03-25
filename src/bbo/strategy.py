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


def recent_instability(y_hist: np.ndarray, window: int = config.W8_INSTABILITY_WINDOW) -> float:
    """Late-stage instability proxy based on recent output volatility."""
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if len(y_hist) < max(window, 3):
        return 0.0
    recent = y_hist[-window:]
    diffs = np.diff(recent)
    if diffs.size == 0:
        return 0.0
    span = max(float(np.max(y_hist) - np.min(y_hist)), 1e-12)
    return float(np.std(diffs) / span)


def choose_strategy(week: int, y_hist: np.ndarray) -> str:
    """Squirrel-inspired switching policy with Week 8 instability awareness.

    - early rounds: broad exploration
    - middle rounds: model-based BO
    - late rounds: local refinement unless stagnating or unstable
    """
    instability = recent_instability(y_hist)
    if week <= 2:
        return "explore"
    if week <= 5:
        return "bo"
    if is_stagnating(y_hist):
        return "explore"
    if len(np.asarray(y_hist).reshape(-1)) >= config.LATE_STAGE_MIN_POINTS and instability >= config.W8_INSTABILITY_THRESHOLD:
        return "bo"
    return "refine"


def choose_acquisition(
    week: int,
    *,
    stagnating: bool = False,
    instability: float = 0.0,
    strategy: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Acquisition portfolio with instability-aware fallback."""
    if stagnating or instability >= config.W8_INSTABILITY_THRESHOLD:
        return "ucb"
    if strategy and strategy.lower().strip() == "refine":
        return "ei"
    if default:
        base = default.lower().strip()
    else:
        cycle = ("ei", "pi", "ucb")
        base = cycle[(max(week, 1) - 1) % len(cycle)]
    return base


def adaptive_exploration_params(week: int, y_hist: np.ndarray, func_idx: int) -> Dict[str, float]:
    """Week 6/8: Sophisticated parameter tuning with adaptive exploration."""
    convergence_info = analyze_convergence(y_hist)
    instability = recent_instability(y_hist)

    # Adaptive exploration decay, stronger in late stage
    late_stage_multiplier = 0.65 if len(np.asarray(y_hist).reshape(-1)) >= config.LATE_STAGE_MIN_POINTS else 1.0
    base_exploration = config.XI_EXPLORE * (config.ADAPTIVE_EXPLORATION_RATE ** week) * late_stage_multiplier

    # Multi-objective function portfolio balancing
    func_weight = config.MULTI_OBJECTIVE_WEIGHTS[func_idx] if func_idx < len(config.MULTI_OBJECTIVE_WEIGHTS) else 1.0

    # Enhanced uncertainty quantification
    uncertainty_factor = config.UNCERTAINTY_BOOST_FACTOR if convergence_info["stability_score"] < 0.5 else 1.0
    uncertainty_factor *= (1.0 + config.W8_SIGMA_BOOST * instability)

    # Sophisticated parameter adaptation
    if convergence_info["improvement_trend"] < config.MIN_IMPROVEMENT_THRESHOLD:
        xi_adaptive = base_exploration * 1.5 * uncertainty_factor
        beta_adaptive = config.BETA_EXPLORE * func_weight * uncertainty_factor
        mode = "adaptive_explore"
    else:
        xi_adaptive = config.XI_EXPLOIT + (base_exploration * 0.3 * uncertainty_factor)
        beta_adaptive = config.BETA_EXPLOIT * func_weight * max(1.0, 1.0 + 0.5 * instability)
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
        "instability": float(instability),
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


def llm_strategy_metadata(*, dim: int, strategy: str, instability: float, n_observations: int) -> Dict[str, float | int | str]:
    """Reflection-aligned metadata proxies for Module 19 write-up."""
    stable = instability < config.W8_INSTABILITY_THRESHOLD and strategy == "refine"
    prompt_pattern = config.W8_PROMPT_PATTERN_STABLE if stable else config.W8_PROMPT_PATTERN_UNSTABLE
    temperature = config.W8_TEMPERATURE_STABLE if stable else config.W8_TEMPERATURE_UNSTABLE
    token_pressure = "moderate" if dim >= config.W8_TOKEN_PRESSURE_RISK_LONG_DIM else "low"
    if n_observations >= config.LATE_STAGE_MIN_POINTS and dim >= config.W8_TOKEN_PRESSURE_RISK_LONG_DIM + 1:
        token_pressure = "elevated"
    return {
        "prompt_pattern": prompt_pattern,
        "temperature": float(temperature),
        "top_p": float(config.W8_TOP_P),
        "top_k": int(config.W8_TOP_K),
        "max_tokens": int(config.W8_MAX_TOKENS),
        "token_pressure_risk": token_pressure,
    }
