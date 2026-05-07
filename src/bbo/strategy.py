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


def drawdown_ratio(y_hist: np.ndarray, window: int = config.W10_DRAWDOWN_WINDOW) -> float:
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if y_hist.size < 2:
        return 0.0
    lookback = y_hist[-window:] if y_hist.size >= window else y_hist
    best_recent = float(np.max(lookback))
    worst_recent = float(np.min(lookback))
    span = max(best_recent - worst_recent, 1e-12)
    return float(np.clip((best_recent - float(y_hist[-1])) / span, 0.0, 1.0))


def recent_trend_score(y_hist: np.ndarray, window: int = config.W10_RECENT_TREND_WINDOW) -> float:
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if y_hist.size < 3:
        return 0.0
    recent = y_hist[-window:] if y_hist.size >= window else y_hist
    if recent.size < 2:
        return 0.0
    return float(np.mean(np.diff(recent)))


def dimension_scaling_pressure(dim: int, n_observations: int) -> float:
    if dim < config.W9_DIMENSION_SCALE_TRIGGER:
        return 0.0
    base = config.W9_DIMENSION_SCALE_BOOST * ((dim - config.W9_DIMENSION_SCALE_TRIGGER + 1) / max(dim, 1))
    if n_observations >= config.LATE_STAGE_MIN_POINTS:
        base *= 1.35
    return float(base)



def pca_variance_structure(X: np.ndarray, y: np.ndarray, *, top_ratio: float = config.W12_PCA_TOP_RATIO) -> Dict[str, float]:
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    if X.ndim != 2 or X.shape[0] < config.W12_PCA_MIN_POINTS or X.shape[1] < 2:
        return {"enabled": 0.0, "n_components": 0.0, "explained_variance": 0.0, "redundancy": 0.0}
    n_top = max(3, int(np.ceil(top_ratio * X.shape[0])))
    top_idx = np.argsort(y)[-n_top:]
    X_top = X[top_idx]
    X_centered = X_top - np.mean(X_top, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(X_centered, full_matrices=False)
    variance = singular_values ** 2
    total = float(np.sum(variance))
    if total <= 1e-12:
        return {"enabled": 0.0, "n_components": 0.0, "explained_variance": 0.0, "redundancy": 1.0}
    ratio = variance / total
    cumulative = np.cumsum(ratio)
    n_components = int(np.searchsorted(cumulative, config.W12_PCA_EXPLAINED_VARIANCE_TARGET) + 1)
    n_components = int(np.clip(n_components, 1, min(config.W12_PCA_MAX_COMPONENTS, X.shape[1])))
    explained = float(cumulative[n_components - 1])
    redundancy = float(np.clip(1.0 - (n_components / max(X.shape[1], 1)), 0.0, 1.0))
    return {
        "enabled": 1.0,
        "n_components": float(n_components),
        "explained_variance": explained,
        "redundancy": redundancy,
    }


def rl_reward_signal(y_hist: np.ndarray, window: int = 4) -> Dict[str, float]:
    """Convert sequential feedback into an RL-style reward summary.

    Positive recent deltas indicate that the current policy is being rewarded;
    drawdowns or flat rewards increase the need for exploration.
    """
    y_hist = np.asarray(y_hist, float).reshape(-1)
    if y_hist.size < 3:
        return {"recent_reward": 0.0, "reward_rate": 0.0, "epsilon": config.W13_EPSILON_BASE}
    recent = y_hist[-window:] if y_hist.size >= window else y_hist
    deltas = np.diff(recent)
    span = max(float(np.max(y_hist) - np.min(y_hist)), 1e-12)
    recent_reward = float(np.mean(deltas) / span) if deltas.size else 0.0
    reward_rate = float((float(y_hist[-1]) - float(y_hist[-2])) / span)
    instability = recent_instability(y_hist)
    drawdown = drawdown_ratio(y_hist)
    epsilon = config.W13_EPSILON_BASE
    epsilon += 0.20 * max(0.0, -recent_reward)
    epsilon += 0.18 * instability
    epsilon += 0.12 * drawdown
    epsilon = float(np.clip(epsilon, config.W13_EPSILON_MIN, config.W13_EPSILON_MAX))
    return {"recent_reward": recent_reward, "reward_rate": reward_rate, "epsilon": epsilon}

def choose_strategy(week: int, y_hist: np.ndarray, *, dim: Optional[int] = None) -> str:
    instability = recent_instability(y_hist)
    emergence = emergence_score(y_hist)
    drawdown = drawdown_ratio(y_hist)
    trend = recent_trend_score(y_hist)
    n_obs = len(np.asarray(y_hist).reshape(-1))
    scaling_pressure = dimension_scaling_pressure(dim or 0, n_obs)
    pca_info = {"enabled": 0.0, "explained_variance": 0.0, "redundancy": 0.0}

    if week <= 2:
        return "explore"
    if week <= 5:
        return "bo"
    if is_stagnating(y_hist):
        return "explore"
    if n_obs >= config.LATE_STAGE_MIN_POINTS and (
        drawdown >= config.W10_DRAWDOWN_TRIGGER
        or trend <= config.W10_RECENT_TREND_TRIGGER
    ):
        return "recover"
    if instability >= config.W8_INSTABILITY_THRESHOLD or scaling_pressure > 0.18:
        return "hedge"
    if n_obs >= config.W13_RL_MIN_POINTS:
        return "rl_feedback"
    if n_obs >= config.W12_PCA_MIN_POINTS and (dim or 0) >= 3:
        return "pca_variance"
    if n_obs >= config.W11_CLUSTER_MIN_POINTS:
        return "cluster"
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
    if strategy and strategy.lower().strip() == "recover":
        return "ei"
    if stagnating or instability >= config.W8_INSTABILITY_THRESHOLD:
        return "ucb"
    if abs(emergence) >= config.W9_EMERGENCE_Z_THRESHOLD and strategy == "hedge":
        return "ucb"
    if strategy and strategy.lower().strip() == "pca_variance":
        return "ei"
    if strategy and strategy.lower().strip() == "rl_feedback":
        return "ucb" if (instability >= 0.12 or abs(emergence) >= config.W9_EMERGENCE_Z_THRESHOLD) else "ei"
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
    drawdown = drawdown_ratio(y_hist)
    trend = recent_trend_score(y_hist)

    n_hist = len(np.asarray(y_hist).reshape(-1))
    rl_signal = rl_reward_signal(y_hist)
    late_stage_multiplier = 0.50 if n_hist >= config.W13_RL_MIN_POINTS else (0.55 if n_hist >= config.W12_PCA_MIN_POINTS else (0.65 if n_hist >= config.LATE_STAGE_MIN_POINTS else 1.0))
    base_exploration = config.XI_EXPLORE * (config.ADAPTIVE_EXPLORATION_RATE ** week) * late_stage_multiplier
    func_weight = config.MULTI_OBJECTIVE_WEIGHTS[func_idx] if func_idx < len(config.MULTI_OBJECTIVE_WEIGHTS) else 1.0

    uncertainty_factor = config.UNCERTAINTY_BOOST_FACTOR if convergence_info["stability_score"] < 0.5 else 1.0
    uncertainty_factor *= (1.0 + config.W8_SIGMA_BOOST * instability)
    uncertainty_factor *= (1.0 + max(0.0, abs(emergence) - config.W9_EMERGENCE_Z_THRESHOLD) * config.W9_EMERGENCE_WEIGHT)
    uncertainty_factor *= (1.0 + scaling_pressure)
    uncertainty_factor *= (1.0 + drawdown)

    if drawdown >= config.W10_DRAWDOWN_TRIGGER or trend <= config.W10_RECENT_TREND_TRIGGER:
        xi_adaptive = base_exploration * 1.2 * uncertainty_factor
        beta_adaptive = config.BETA_EXPLORE * func_weight * max(1.0, uncertainty_factor)
        mode = "adaptive_recover"
    elif n_hist >= config.W13_RL_MIN_POINTS:
        xi_adaptive = config.XI_EXPLOIT + base_exploration * (0.5 + rl_signal["epsilon"]) * uncertainty_factor
        beta_adaptive = config.BETA_EXPLOIT * func_weight * max(1.0, 1.0 + rl_signal["epsilon"] + instability + scaling_pressure)
        mode = "rl_feedback_adaptive"
    elif convergence_info["improvement_trend"] < config.MIN_IMPROVEMENT_THRESHOLD:
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
        "drawdown_ratio": float(drawdown),
        "recent_trend_score": float(trend),
        "rl_recent_reward": float(rl_signal["recent_reward"]),
        "rl_reward_rate": float(rl_signal["reward_rate"]),
        "rl_epsilon": float(rl_signal["epsilon"]),
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
    stable = instability < config.W8_INSTABILITY_THRESHOLD and strategy in ("refine", "recover") and abs(emergence) < config.W9_EMERGENCE_Z_THRESHOLD
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
