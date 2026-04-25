from __future__ import annotations
from typing import Any, List, Tuple, Optional
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
import warnings
from . import config

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _topk_center(X: np.ndarray, y: np.ndarray, dim: int) -> np.ndarray:
    top_k = config.W10_TOPK_CENTER_LOW_D if dim <= 4 else config.W10_TOPK_CENTER_HIGH_D
    top_k = max(1, min(top_k, X.shape[0]))
    idx = np.argsort(y)[-top_k:]
    return np.mean(np.asarray(X[idx], float), axis=0)


def _cluster_count(n_points: int) -> int:
    if n_points < config.W11_CLUSTER_MIN_POINTS:
        return 0
    return int(np.clip(round(np.sqrt(n_points / 2.0)), config.W11_CLUSTER_MIN_K, config.W11_CLUSTER_MAX_K))


def _cluster_summary(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    n_points, dim = X.shape
    k = _cluster_count(n_points)
    if k <= 1:
        return {"enabled": False}
    try:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    except Exception:
        return {"enabled": False}

    clusters = []
    y_span = max(float(np.max(y) - np.min(y)), 1e-12)
    for label in range(k):
        mask = labels == label
        if not np.any(mask):
            continue
        Xc = X[mask]
        yc = y[mask]
        centroid = np.mean(Xc, axis=0)
        spread = np.std(Xc, axis=0)
        best_local_idx = int(np.argmax(yc))
        best_x = Xc[best_local_idx]
        quality = (float(np.max(yc)) - float(np.min(y))) / y_span
        compactness = 1.0 / (1.0 + float(np.mean(np.linalg.norm(Xc - centroid, axis=1))))
        clusters.append({
            "label": int(label),
            "size": int(Xc.shape[0]),
            "centroid": centroid,
            "spread": spread,
            "best_x": best_x,
            "best_y": float(np.max(yc)),
            "mean_y": float(np.mean(yc)),
            "quality": float(quality),
            "compactness": float(compactness),
        })
    if not clusters:
        return {"enabled": False}
    clusters.sort(key=lambda c: (c["best_y"], c["mean_y"], c["compactness"]), reverse=True)
    return {"enabled": True, "k": int(k), "labels": labels, "clusters": clusters, "best_cluster": clusters[0]}


def _cluster_candidate_cloud(rng: np.random.Generator, X: np.ndarray, y: np.ndarray, dim: int, n_candidates: int, seed: int) -> tuple[np.ndarray, dict]:
    summary = _cluster_summary(X, y, seed)
    if not summary.get("enabled"):
        return rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_candidates, dim)), summary

    best = summary["best_cluster"]
    cap = config.W11_CLUSTER_SPREAD_CAP_LOW_D if dim <= 4 else config.W11_CLUSTER_SPREAD_CAP_HIGH_D
    spread = np.clip(best["spread"] * config.W11_CLUSTER_SCALE_SHRINK, config.W11_CLUSTER_SPREAD_FLOOR, cap)
    n_top = max(1, int(config.W11_CLUSTER_TOP_RATIO * n_candidates))
    n_boundary = max(1, int(config.W11_CLUSTER_BOUNDARY_RATIO * n_candidates))
    n_global = max(1, n_candidates - n_top - n_boundary)

    centroid_cloud = rng.normal(loc=best["centroid"], scale=spread, size=(n_top // 2, dim))
    best_cloud = rng.normal(loc=best["best_x"], scale=spread * 0.80, size=(n_top - (n_top // 2), dim))

    boundary_centers = []
    for c in summary["clusters"][1:]:
        boundary_centers.append(0.5 * (best["centroid"] + c["centroid"]))
    if boundary_centers:
        centers = np.asarray(boundary_centers, float)
        picks = centers[rng.integers(0, centers.shape[0], size=n_boundary)]
        boundary_cloud = rng.normal(loc=picks, scale=spread * 1.15, size=(n_boundary, dim))
    else:
        boundary_cloud = rng.normal(loc=best["centroid"], scale=spread * 1.4, size=(n_boundary, dim))

    global_cloud = rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_global, dim))
    Xcand = np.vstack([centroid_cloud, best_cloud, boundary_cloud, global_cloud])
    return np.clip(Xcand, config.W9_MIN_BOUND, config.W9_MAX_BOUND), summary

def _trust_region_scale(dim: int, strategy: str, drawdown_ratio: float) -> float:
    base = config.W10_RECOVER_SCALE_LOW_D if dim <= 4 else config.W10_RECOVER_SCALE_HIGH_D
    scale = base * (1.0 - 0.35 * float(np.clip(drawdown_ratio, 0.0, 1.0)))
    if strategy == "refine":
        scale *= config.W10_REFINE_TRUST_SHRINK
    elif strategy in ("hedge", "bo"):
        scale *= config.W10_HEDGE_TRUST_EXPAND
    return float(np.clip(scale, config.W10_TRUST_REGION_MIN, config.W10_TRUST_REGION_MAX))


def _acq_ei(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - y_best - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def _acq_pi(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - y_best - xi) / sigma
    return norm.cdf(z)


def _acq_ucb(mu: np.ndarray, sigma: np.ndarray, beta: float) -> np.ndarray:
    return mu + beta * sigma


def _restarts(dim: int, n_points: int) -> int:
    if dim >= config.W9_FAST_GP_DIM_THRESHOLD and n_points >= config.W9_FAST_GP_POINT_THRESHOLD:
        return config.RESTARTS_FAST_HIGH_D
    if dim <= 3:
        return config.RESTARTS_LOW_D
    if dim <= 5:
        return config.RESTARTS_MID_D
    return config.RESTARTS_HIGH_D


def _kernel_pool(dim: int) -> List[Any]:
    ls0 = 0.30 if dim <= 3 else (0.40 if dim <= 5 else 0.60)
    base = [
        RBF(length_scale=np.ones(dim) * ls0, length_scale_bounds=config.LENGTH_SCALE_BOUNDS),
        Matern(length_scale=np.ones(dim) * ls0, length_scale_bounds=config.LENGTH_SCALE_BOUNDS, nu=1.5),
        Matern(length_scale=np.ones(dim) * ls0, length_scale_bounds=config.LENGTH_SCALE_BOUNDS, nu=2.5),
    ]
    pool = []
    for bk in base:
        pool.append(
            ConstantKernel(1.0, (1e-3, 1e3)) * bk
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=config.WHITE_NOISE_BOUNDS)
        )
    return pool


def fit_best_gp_by_lml(X: np.ndarray, y: np.ndarray, dim: int, seed: int):
    best_gp: Optional[GaussianProcessRegressor] = None
    best_lml = -np.inf
    details: List[Tuple[str, float]] = []
    n_restarts = _restarts(dim, len(y))
    for j, k in enumerate(_kernel_pool(dim)):
        gp = GaussianProcessRegressor(
            kernel=k,
            alpha=config.NOISE_ALPHA,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=seed + 17 * j,
        )
        gp.fit(X, y)
        lml = float(gp.log_marginal_likelihood(gp.kernel_.theta))
        details.append((str(gp.kernel_), lml))
        if lml > best_lml:
            best_lml = lml
            best_gp = gp
    assert best_gp is not None
    return best_gp, best_lml, details


def _build_candidates(
    rng: np.random.Generator,
    *,
    X: np.ndarray,
    y: np.ndarray,
    dim: int,
    n_candidates: int,
    strategy: str,
    drawdown_ratio: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    strategy = strategy.lower().strip()
    if strategy == "explore":
        return rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_candidates, dim)), {"enabled": False}

    if strategy == "cluster":
        return _cluster_candidate_cloud(rng, X, y, dim, n_candidates, seed)

    best_x = np.asarray(X[int(np.argmax(y))], dtype=float)
    center_x = _topk_center(X, y, dim)

    if strategy == "recover":
        trust_scale = _trust_region_scale(dim, strategy, drawdown_ratio)
        n_best = max(1, int(config.W10_RECOVER_BEST_RATIO * n_candidates))
        n_center = max(1, int(config.W10_RECOVER_CENTER_RATIO * n_candidates))
        n_global = max(1, n_candidates - n_best - n_center)
        around_best = np.clip(rng.normal(loc=best_x, scale=trust_scale, size=(n_best, dim)), config.W9_MIN_BOUND, config.W9_MAX_BOUND)
        around_center = np.clip(rng.normal(loc=center_x, scale=trust_scale * 1.15, size=(n_center, dim)), config.W9_MIN_BOUND, config.W9_MAX_BOUND)
        global_ = rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_global, dim))
        return np.vstack([around_best, around_center, global_]), {"enabled": False}

    if strategy == "refine":
        n_local = max(1, int(config.W8_LOCAL_CANDIDATE_RATIO * n_candidates))
        n_global = max(1, n_candidates - n_local)
        local_scale = _trust_region_scale(dim, strategy, drawdown_ratio)
        local = np.clip(rng.normal(loc=best_x, scale=local_scale, size=(n_local, dim)), config.W9_MIN_BOUND, config.W9_MAX_BOUND)
        global_ = rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_global, dim))
        return np.vstack([local, global_]), {"enabled": False}

    if strategy == "hedge":
        n_local = max(1, int(config.W8_HEDGE_LOCAL_RATIO * n_candidates))
        n_global = max(1, n_candidates - n_local)
        local_scale = _trust_region_scale(dim, strategy, drawdown_ratio) * 1.10
        local = np.clip(rng.normal(loc=best_x, scale=local_scale, size=(n_local, dim)), config.W9_MIN_BOUND, config.W9_MAX_BOUND)
        global_ = rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_global, dim))
        return np.vstack([local, global_]), {"enabled": False}

    n_local = max(1, int(0.55 * n_candidates))
    n_global = max(1, n_candidates - n_local)
    local_scale = _trust_region_scale(dim, strategy, drawdown_ratio) * 1.20
    local = np.clip(rng.normal(loc=best_x, scale=local_scale, size=(n_local, dim)), config.W9_MIN_BOUND, config.W9_MAX_BOUND)
    global_ = rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_global, dim))
    return np.vstack([local, global_]), {"enabled": False}


def _boundary_penalty(Xcand: np.ndarray, margin: float = config.W8_BOUNDARY_MARGIN) -> np.ndarray:
    low = np.clip((margin - Xcand) / margin, 0.0, None)
    high = np.clip((Xcand - (1.0 - margin)) / margin, 0.0, None)
    return np.mean(low + high, axis=1)


def _repeat_penalty(Xcand: np.ndarray, X_hist: np.ndarray, repeat_distance: float = config.W8_REPEAT_DISTANCE) -> np.ndarray:
    if X_hist.size == 0:
        return np.zeros(Xcand.shape[0], dtype=float)
    dists = np.sqrt(((Xcand[:, None, :] - X_hist[None, :, :]) ** 2).sum(axis=2))
    min_dist = dists.min(axis=1)
    return np.clip((repeat_distance - min_dist) / repeat_distance, 0.0, 1.0)


def _ruggedness_penalty(Xcand: np.ndarray, X_hist: np.ndarray, y_hist: np.ndarray) -> np.ndarray:
    if len(y_hist) < 5 or X_hist.shape[0] < 5:
        return np.zeros(Xcand.shape[0], dtype=float)
    top_k = min(5, X_hist.shape[0])
    top_idx = np.argsort(y_hist)[-top_k:]
    anchors = X_hist[top_idx]
    dists = np.sqrt(((Xcand[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2))
    nearest = dists.min(axis=1)
    return np.exp(-nearest / max(0.05 * Xcand.shape[1], 1e-6))




def _consensus_bonus(Xcand: np.ndarray, X_hist: np.ndarray, y_hist: np.ndarray) -> np.ndarray:
    if X_hist.shape[0] < 3:
        return np.zeros(Xcand.shape[0], dtype=float)
    center = _topk_center(X_hist, y_hist, X_hist.shape[1])
    dist = np.sqrt(((Xcand - center) ** 2).sum(axis=1))
    scale = max(0.08 * Xcand.shape[1], 1e-6)
    return np.exp(-dist / scale)


def _last_point_repulsion(Xcand: np.ndarray, last_x: np.ndarray, trigger: float) -> np.ndarray:
    if trigger <= 0.0:
        return np.zeros(Xcand.shape[0], dtype=float)
    dist = np.sqrt(((Xcand - last_x.reshape(1, -1)) ** 2).sum(axis=1))
    return np.clip((config.W10_LAST_POINT_REPULSION_DISTANCE - dist) / config.W10_LAST_POINT_REPULSION_DISTANCE, 0.0, 1.0) * float(trigger)

def propose_next_point(
    X: np.ndarray,
    y: np.ndarray,
    *,
    acquisition: str,
    xi: float,
    beta: float,
    seed: int,
    n_candidates: int,
    strategy: str = "bo",
    instability: float = 0.0,
    emergence_score: float = 0.0,
    ruggedness_score: float = 0.0,
    dimension_scaling_pressure: float = 0.0,
    drawdown_ratio: float = 0.0,
):
    rng = np.random.default_rng(seed)
    dim = X.shape[1]
    gp, best_lml, lml_details = fit_best_gp_by_lml(X, y, dim=dim, seed=seed)

    Xcand, cluster_info = _build_candidates(rng, X=X, y=y, dim=dim, n_candidates=n_candidates, strategy=strategy, drawdown_ratio=drawdown_ratio, seed=seed)
    mu, sigma = gp.predict(Xcand, return_std=True)
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    y_best = float(np.max(y))

    a = acquisition.lower().strip()
    if a == "ei":
        raw_score = _acq_ei(mu, sigma, y_best, xi)
    elif a == "pi":
        raw_score = _acq_pi(mu, sigma, y_best, xi)
    elif a == "ucb":
        raw_score = _acq_ucb(mu, sigma, beta)
    else:
        raise ValueError("ACQUISITION must be one of: ei, pi, ucb")

    sigma_bonus = sigma * (config.W8_SIGMA_BOOST * float(instability) + float(dimension_scaling_pressure))
    emergence_bonus = sigma * max(0.0, abs(float(emergence_score)) - config.W9_EMERGENCE_Z_THRESHOLD) * config.W9_EMERGENCE_WEIGHT
    boundary_pen = _boundary_penalty(Xcand)
    repeat_pen = _repeat_penalty(Xcand, X)
    rugged_pen = _ruggedness_penalty(Xcand, X, y)
    consensus_bonus = _consensus_bonus(Xcand, X, y)
    last_point_pen = _last_point_repulsion(Xcand, np.asarray(X[-1], float), max(0.0, float(drawdown_ratio) - config.W10_DRAWDOWN_TRIGGER))
    if cluster_info.get("enabled"):
        best_cluster = cluster_info["best_cluster"]
        dist_to_cluster = np.sqrt(((Xcand - best_cluster["centroid"]) ** 2).sum(axis=1))
        cluster_scale = max(0.08 * dim, 1e-6)
        cluster_quality_bonus = best_cluster["quality"] * np.exp(-dist_to_cluster / cluster_scale)
        cluster_diversity_bonus = np.zeros(Xcand.shape[0], dtype=float)
        for c in cluster_info["clusters"][1:]:
            d = np.sqrt(((Xcand - c["centroid"]) ** 2).sum(axis=1))
            cluster_diversity_bonus = np.maximum(cluster_diversity_bonus, np.exp(-d / cluster_scale))
    else:
        cluster_quality_bonus = np.zeros(Xcand.shape[0], dtype=float)
        cluster_diversity_bonus = np.zeros(Xcand.shape[0], dtype=float)
    score = (
        raw_score
        + sigma_bonus
        + emergence_bonus
        + config.W10_CONSENSUS_BONUS_WEIGHT * consensus_bonus
        + config.W11_CLUSTER_QUALITY_BONUS_WEIGHT * cluster_quality_bonus
        + config.W11_CLUSTER_DIVERSITY_BONUS_WEIGHT * cluster_diversity_bonus
        - config.W8_BOUNDARY_PENALTY_WEIGHT * boundary_pen
        - config.W8_REPEAT_PENALTY_WEIGHT * repeat_pen
        - config.W9_RUGGEDNESS_PENALTY_WEIGHT * float(ruggedness_score) * rugged_pen
        - config.W10_LAST_POINT_REPULSION_WEIGHT * last_point_pen
    )

    idx = int(np.argmax(score))
    return Xcand[idx], {
        "kernel": str(gp.kernel_),
        "best_lml": float(best_lml),
        "lml_candidates": [(k, float(v)) for k, v in lml_details],
        "mu_at_choice": float(mu[idx]),
        "sigma_at_choice": float(sigma[idx]),
        "strategy": strategy,
        "acquisition_used": a,
        "instability": float(instability),
        "emergence_score": float(emergence_score),
        "ruggedness_score": float(ruggedness_score),
        "dimension_scaling_pressure": float(dimension_scaling_pressure),
        "drawdown_ratio": float(drawdown_ratio),
        "raw_score_at_choice": float(raw_score[idx]),
        "adjusted_score_at_choice": float(score[idx]),
        "sigma_bonus_at_choice": float(sigma_bonus[idx]),
        "emergence_bonus_at_choice": float(emergence_bonus[idx]),
        "boundary_penalty_at_choice": float(boundary_pen[idx]),
        "repeat_penalty_at_choice": float(repeat_pen[idx]),
        "ruggedness_penalty_at_choice": float(rugged_pen[idx]),
        "consensus_bonus_at_choice": float(consensus_bonus[idx]),
        "last_point_repulsion_at_choice": float(last_point_pen[idx]),
        "cluster_enabled": bool(cluster_info.get("enabled", False)),
        "cluster_count": int(cluster_info.get("k", 0) or 0),
        "cluster_quality_bonus_at_choice": float(cluster_quality_bonus[idx]),
        "cluster_diversity_bonus_at_choice": float(cluster_diversity_bonus[idx]),
        "cluster_target_label": int(cluster_info.get("best_cluster", {}).get("label", -1)) if cluster_info.get("enabled") else -1,
        "fast_gp_mode": bool(dim >= config.W9_FAST_GP_DIM_THRESHOLD and len(y) >= config.W9_FAST_GP_POINT_THRESHOLD),
    }
