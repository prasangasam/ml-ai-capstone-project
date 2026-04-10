from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from . import config


def decide_mode_maximise(y_last: float, y_hist: np.ndarray, tol_frac: float = config.EXPLOIT_TOL_FRAC_OF_RANGE) -> str:
    y_hist = np.asarray(y_hist, float).reshape(-1)
    best = float(np.max(y_hist))
    worst = float(np.min(y_hist))
    span = max(best - worst, 1e-12)
    dist_frac = (best - float(y_last)) / span
    return "exploit" if dist_frac <= tol_frac else "explore"


def analyze_convergence(y_hist: np.ndarray, window: int = config.CONVERGENCE_WINDOW) -> Dict[str, float]:
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
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if len(y_hist) < window + 1:
        return False
    recent = y_hist[-window:]
    return bool((float(np.max(recent)) - float(np.min(recent))) < tol)


def recent_instability(y_hist: np.ndarray, window: int = config.W8_INSTABILITY_WINDOW) -> float:
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if len(y_hist) < max(window, 3):
        return 0.0
    recent = y_hist[-window:]
    diffs = np.diff(recent)
    if diffs.size == 0:
        return 0.0
    span = max(float(np.max(y_hist) - np.min(y_hist)), 1e-12)
    return float(np.std(diffs) / span)


def emergence_score(y_hist: np.ndarray) -> float:
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if len(y_hist) < 6:
        return 0.0
    earlier = y_hist[:-1]
    last = float(y_hist[-1])
    mu = float(np.mean(earlier))
    sigma = float(np.std(earlier))
    if sigma < 1e-12:
        return 0.0
    return float((last - mu) / sigma)


def ruggedness_score(y_hist: np.ndarray, window: int = config.W9_RUGGEDNESS_WINDOW) -> float:
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if len(y_hist) < max(window, 4):
        return 0.0
    recent = y_hist[-window:]
    first_diff = np.diff(recent)
    second_diff = np.diff(first_diff)
    denom = max(float(np.max(y_hist) - np.min(y_hist)), 1e-12)
    if second_diff.size == 0:
        return 0.0
    return float(np.mean(np.abs(second_diff)) / denom)


def dimension_scaling_pressure(dim: int, n_observations: int) -> float:
    if dim < config.W9_DIMENSION_SCALE_TRIGGER:
        return 0.0
    base = config.W9_DIMENSION_SCALE_BOOST * ((dim - config.W9_DIMENSION_SCALE_TRIGGER + 1) / max(dim, 1))
    if n_observations >= config.LATE_STAGE_MIN_POINTS:
        base *= 1.35
    return float(base)


def choose_strategy(week: int, y_hist: np.ndarray, *, dim: Optional[int] = None) -> str:
    instability = recent_instability(y_hist)
    emergence = emergence_score(y_hist)
    n_obs = len(np.asarray(y_hist).reshape(-1))
    scaling_pressure = dimension_scaling_pressure(dim or 0, n_obs)

    if week <= 2:
        return "explore"
    if week <= 5:
        return "bo"
    if is_stagnating(y_hist):
        return "explore"
    if instability >= config.W8_INSTABILITY_THRESHOLD or scaling_pressure > 0.18:
        return "hedge"
    if n_obs >= config.LATE_STAGE_MIN_POINTS and abs(emergence) >= config.W9_EMERGENCE_Z_THRESHOLD:
        return "hedge"
    if n_obs >= config.LATE_STAGE_MIN_POINTS and instability >= 0.10:
        return "bo"
    return "refine"


def choose_acquisition(
    week: int,
    *,
    stagnating: bool = False,
    instability: float = 0.0,
    strategy: Optional[str] = None,
    default: Optional[str] = None,
    emergence: float = 0.0,
) -> str:
    if stagnating or instability >= config.W8_INSTABILITY_THRESHOLD:
        return "ucb"
    if abs(emergence) >= config.W9_EMERGENCE_Z_THRESHOLD and strategy == "hedge":
        return "ucb"
    if strategy and strategy.lower().strip() == "refine":
        return "ei"
    if strategy and strategy.lower().strip() == "hedge":
        return "ei" if abs(emergence) < config.W9_EMERGENCE_Z_THRESHOLD else "ucb"
    if default:
        base = default.lower().strip()
    else:
        cycle = ("ei", "pi", "ucb")
        base = cycle[(max(week, 1) - 1) % len(cycle)]
    return base


def adaptive_exploration_params(week: int, y_hist: np.ndarray, func_idx: int, dim: Optional[int] = None) -> Dict[str, float]:
    convergence_info = analyze_convergence(y_hist)
    instability = recent_instability(y_hist)
    emergence = emergence_score(y_hist)
    ruggedness = ruggedness_score(y_hist)
    scaling_pressure = dimension_scaling_pressure(dim or 0, len(np.asarray(y_hist).reshape(-1)))

    late_stage_multiplier = 0.65 if len(np.asarray(y_hist).reshape(-1)) >= config.LATE_STAGE_MIN_POINTS else 1.0
    base_exploration = config.XI_EXPLORE * (config.ADAPTIVE_EXPLORATION_RATE ** week) * late_stage_multiplier
    func_weight = config.MULTI_OBJECTIVE_WEIGHTS[func_idx] if func_idx < len(config.MULTI_OBJECTIVE_WEIGHTS) else 1.0

    uncertainty_factor = config.UNCERTAINTY_BOOST_FACTOR if convergence_info["stability_score"] < 0.5 else 1.0
    uncertainty_factor *= (1.0 + config.W8_SIGMA_BOOST * instability)
    uncertainty_factor *= (1.0 + max(0.0, abs(emergence) - config.W9_EMERGENCE_Z_THRESHOLD) * config.W9_EMERGENCE_WEIGHT)
    uncertainty_factor *= (1.0 + scaling_pressure)

    if convergence_info["improvement_trend"] < config.MIN_IMPROVEMENT_THRESHOLD:
        xi_adaptive = base_exploration * 1.5 * uncertainty_factor
        beta_adaptive = config.BETA_EXPLORE * func_weight * uncertainty_factor
        mode = "adaptive_explore"
    else:
        xi_adaptive = config.XI_EXPLOIT + (base_exploration * 0.3 * uncertainty_factor)
        beta_adaptive = config.BETA_EXPLOIT * func_weight * max(1.0, 1.0 + 0.5 * instability + scaling_pressure)
        mode = "adaptive_exploit"

    beta_adaptive *= max(0.75, 1.0 - 0.25 * ruggedness)
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
        "emergence_score": float(emergence),
        "ruggedness_score": float(ruggedness),
        "dimension_scaling_pressure": float(scaling_pressure),
    }


def tune_params(mode: str, acquisition: str, week: Optional[int] = None, y_hist: Optional[np.ndarray] = None, func_idx: Optional[int] = None, dim: Optional[int] = None) -> Dict[str, float]:
    a = acquisition.lower().strip()
    if week is not None and y_hist is not None and func_idx is not None and week >= 6:
        return adaptive_exploration_params(week, y_hist, func_idx, dim=dim)
    if a in ("ei", "pi"):
        return {"xi": config.XI_EXPLOIT if mode == "exploit" else config.XI_EXPLORE}
    if a == "ucb":
        return {"beta": config.BETA_EXPLOIT if mode == "exploit" else config.BETA_EXPLORE}
    raise ValueError("ACQUISITION must be one of: ei, pi, ucb")


def multi_objective_portfolio_balance(all_func_performances: List[float]) -> Dict[int, float]:
    performances = np.array(all_func_performances, dtype=float)
    if performances.size == 0:
        return {}
    span = float(performances.max() - performances.min())
    if span <= 1e-12:
        return {i: 1.0 for i in range(len(performances))}

    normalized_perf = (performances - performances.min()) / span
    inverse_weights = 1.0 - normalized_perf
    balanced_weights = inverse_weights / inverse_weights.sum() * len(performances)
    return {i: float(weight) for i, weight in enumerate(balanced_weights)}


def llm_strategy_metadata(*, dim: int, strategy: str, instability: float, n_observations: int, emergence: float = 0.0) -> Dict[str, float | int | str]:
    stable = instability < config.W8_INSTABILITY_THRESHOLD and strategy == "refine" and abs(emergence) < config.W9_EMERGENCE_Z_THRESHOLD
    prompt_pattern = config.W8_PROMPT_PATTERN_STABLE if stable else config.W8_PROMPT_PATTERN_UNSTABLE
    temperature = config.W8_TEMPERATURE_STABLE if stable else config.W8_TEMPERATURE_UNSTABLE
    token_pressure = "moderate" if dim >= config.W8_TOKEN_PRESSURE_RISK_LONG_DIM else "low"
    if n_observations >= config.LATE_STAGE_MIN_POINTS and dim >= config.W8_TOKEN_PRESSURE_RISK_LONG_DIM + 1:
        token_pressure = "elevated"
    if dim >= config.W9_FAST_GP_DIM_THRESHOLD:
        token_pressure = "high"
    return {
        "prompt_pattern": prompt_pattern,
        "temperature": float(temperature),
        "top_p": float(config.W8_TOP_P),
        "top_k": int(config.W8_TOP_K),
        "max_tokens": int(config.W8_MAX_TOKENS),
        "token_pressure_risk": token_pressure,
    }
