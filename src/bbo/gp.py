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


def _pca_summary(X: np.ndarray, y: np.ndarray) -> dict:
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    if X.shape[0] < config.W12_PCA_MIN_POINTS or X.shape[1] < 2:
        return {"enabled": False}
    n_top = max(3, int(np.ceil(config.W12_PCA_TOP_RATIO * X.shape[0])))
    top_idx = np.argsort(y)[-n_top:]
    X_top = X[top_idx]
    center = np.mean(X_top, axis=0)
    X_centered = X_top - center
    try:
        _, singular_values, vh = np.linalg.svd(X_centered, full_matrices=False)
    except Exception:
        return {"enabled": False}
    variance = singular_values ** 2
    total = float(np.sum(variance))
    if total <= 1e-12:
        return {"enabled": False}
    ratio = variance / total
    cumulative = np.cumsum(ratio)
    n_components = int(np.searchsorted(cumulative, config.W12_PCA_EXPLAINED_VARIANCE_TARGET) + 1)
    n_components = int(np.clip(n_components, 1, min(config.W12_PCA_MAX_COMPONENTS, X.shape[1], vh.shape[0])))
    components = vh[:n_components]
    explained = float(cumulative[n_components - 1])
    redundancy = float(np.clip(1.0 - n_components / max(X.shape[1], 1), 0.0, 1.0))
    scale_base = config.W12_PCA_BASE_SCALE_LOW_D if X.shape[1] <= 4 else config.W12_PCA_BASE_SCALE_HIGH_D
    component_scales = np.sqrt(np.maximum(ratio[:n_components], 1e-8))
    component_scales = scale_base * component_scales / max(float(component_scales.max()), 1e-8)
    return {
        "enabled": True,
        "center": center,
        "components": components,
        "component_scales": component_scales,
        "n_components": n_components,
        "explained_variance": explained,
        "redundancy": redundancy,
        "best_x": np.asarray(X[int(np.argmax(y))], float),
    }


def _pca_candidate_cloud(rng: np.random.Generator, X: np.ndarray, y: np.ndarray, dim: int, n_candidates: int) -> tuple[np.ndarray, dict]:
    summary = _pca_summary(X, y)
    if not summary.get("enabled"):
        return _cluster_candidate_cloud(rng, X, y, dim, n_candidates, seed=0)
    n_local = max(1, int(config.W12_PCA_LOCAL_RATIO * n_candidates))
    n_axis = max(1, int(config.W12_PCA_AXIS_RATIO * n_candidates))
    n_global = max(1, n_candidates - n_local - n_axis)
    k = summary["n_components"]
    comps = summary["components"]
    scales = summary["component_scales"]
    center = 0.60 * summary["best_x"] + 0.40 * summary["center"]

    z = rng.normal(size=(n_local, k)) * scales.reshape(1, -1)
    local = center.reshape(1, -1) + z @ comps
    orth_noise = rng.normal(scale=(np.mean(scales) * config.W12_PCA_ORTHOGONAL_SHRINK), size=(n_local, dim))
    local = local + orth_noise

    axes = rng.integers(0, k, size=n_axis)
    signs = rng.choice([-1.0, 1.0], size=n_axis)
    steps = rng.normal(loc=1.0, scale=0.45, size=n_axis) * scales[axes] * signs
    axis_cloud = center.reshape(1, -1) + comps[axes] * steps.reshape(-1, 1)

    global_ = rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_global, dim))
    Xcand = np.vstack([local, axis_cloud, global_])
    return np.clip(Xcand, config.W9_MIN_BOUND, config.W9_MAX_BOUND), summary


def _rl_feedback_summary(y: np.ndarray) -> dict:
    y = np.asarray(y, float).reshape(-1)
    if y.size < 3:
        return {"epsilon": config.W13_EPSILON_BASE, "recent_reward": 0.0, "reward_rate": 0.0, "credit_x_index": int(np.argmax(y))}
    span = max(float(np.max(y) - np.min(y)), 1e-12)
    recent = y[-4:] if y.size >= 4 else y
    deltas = np.diff(recent)
    recent_reward = float(np.mean(deltas) / span) if deltas.size else 0.0
    reward_rate = float((float(y[-1]) - float(y[-2])) / span)
    dd = float(np.clip((float(np.max(y[-5:])) - float(y[-1])) / max(float(np.max(y[-5:]) - np.min(y[-5:])), 1e-12), 0.0, 1.0)) if y.size >= 5 else 0.0
    epsilon = config.W13_EPSILON_BASE + 0.20 * max(0.0, -recent_reward) + 0.12 * dd
    epsilon = float(np.clip(epsilon, config.W13_EPSILON_MIN, config.W13_EPSILON_MAX))
    # Credit assignment proxy: pick the recent point with the highest positive delta.
    if y.size >= 2:
        all_deltas = np.diff(y)
        credit_idx = int(np.argmax(all_deltas) + 1)
        if float(all_deltas[credit_idx - 1]) <= 0.0:
            credit_idx = int(np.argmax(y))
    else:
        credit_idx = int(np.argmax(y))
    return {"epsilon": epsilon, "recent_reward": recent_reward, "reward_rate": reward_rate, "credit_x_index": credit_idx}


def _rl_feedback_candidate_cloud(rng: np.random.Generator, X: np.ndarray, y: np.ndarray, dim: int, n_candidates: int) -> tuple[np.ndarray, dict]:
    summary = _rl_feedback_summary(y)
    best_x = np.asarray(X[int(np.argmax(y))], dtype=float)
    credit_x = np.asarray(X[int(summary["credit_x_index"])], dtype=float)
    center_x = _topk_center(X, y, dim)
    eps = float(summary["epsilon"])
    n_global = max(1, int(max(config.W13_RL_GLOBAL_RATIO, eps) * n_candidates))
    n_credit = max(1, int(config.W13_RL_CREDIT_RATIO * n_candidates))
    n_best = max(1, n_candidates - n_global - n_credit)
    scale = config.W13_RL_LOCAL_SCALE_LOW_D if dim <= 4 else config.W13_RL_LOCAL_SCALE_HIGH_D
    if summary["recent_reward"] < 0.0:
        scale *= 1.25
    best_anchor = 0.70 * best_x + 0.30 * center_x
    best_cloud = rng.normal(loc=best_anchor, scale=scale, size=(n_best, dim))
    credit_cloud = rng.normal(loc=credit_x, scale=scale * 1.15, size=(n_credit, dim))
    global_cloud = rng.uniform(config.W9_MIN_BOUND, config.W9_MAX_BOUND, size=(n_global, dim))
    Xcand = np.vstack([best_cloud, credit_cloud, global_cloud])
    return np.clip(Xcand, config.W9_MIN_BOUND, config.W9_MAX_BOUND), {"enabled": True, "rl_feedback": True, **summary, "credit_x": credit_x}

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
    fast_late_stage = bool(dim >= config.W13_FAST_GP_DIM_THRESHOLD and len(y) >= config.W13_FAST_GP_POINT_THRESHOLD)
    n_restarts = 0 if fast_late_stage else _restarts(dim, len(y))
    kernel_pool = _kernel_pool(dim)[:1] if fast_late_stage else _kernel_pool(dim)
    for j, k in enumerate(kernel_pool):
        gp = GaussianProcessRegressor(
            kernel=k,
            alpha=config.NOISE_ALPHA,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            optimizer=None if fast_late_stage else "fmin_l_bfgs_b",
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

    if strategy == "pca_variance":
        return _pca_candidate_cloud(rng, X, y, dim, n_candidates)

    if strategy == "rl_feedback":
        return _rl_feedback_candidate_cloud(rng, X, y, dim, n_candidates)

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
    recent_trend_score: float = 0.0,
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
    if cluster_info.get("enabled") and "best_cluster" in cluster_info:
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

    if cluster_info.get("enabled") and "components" in cluster_info:
        centered = Xcand - cluster_info["center"].reshape(1, -1)
        projected = centered @ cluster_info["components"].T
        reconstructed = projected @ cluster_info["components"]
        residual = np.sqrt(np.sum((centered - reconstructed) ** 2, axis=1))
        pca_scale = max(float(np.mean(cluster_info["component_scales"])) * dim, 1e-6)
        pca_variance_bonus = cluster_info["explained_variance"] * np.exp(-residual / pca_scale)
        pca_redundancy_penalty = cluster_info["redundancy"] * np.clip(residual / pca_scale, 0.0, 1.0)
    else:
        pca_variance_bonus = np.zeros(Xcand.shape[0], dtype=float)
        pca_redundancy_penalty = np.zeros(Xcand.shape[0], dtype=float)

    if cluster_info.get("rl_feedback"):
        credit_x = np.asarray(cluster_info.get("credit_x"), dtype=float).reshape(1, -1)
        credit_dist = np.sqrt(((Xcand - credit_x) ** 2).sum(axis=1))
        credit_scale = max((config.W13_RL_LOCAL_SCALE_LOW_D if dim <= 4 else config.W13_RL_LOCAL_SCALE_HIGH_D) * dim, 1e-6)
        rl_credit_bonus = np.exp(-credit_dist / credit_scale) * max(0.0, 1.0 + float(cluster_info.get("recent_reward", 0.0)))
        success_bonus = np.maximum(0.0, mu - np.percentile(mu, 70)) / max(float(np.ptp(mu)), 1e-12)
        exploration_bonus = sigma / max(float(np.max(sigma)), 1e-12) * float(cluster_info.get("epsilon", config.W13_EPSILON_BASE))
    else:
        rl_credit_bonus = np.zeros(Xcand.shape[0], dtype=float)
        success_bonus = np.zeros(Xcand.shape[0], dtype=float)
        exploration_bonus = np.zeros(Xcand.shape[0], dtype=float)
    score = (
        raw_score
        + sigma_bonus
        + emergence_bonus
        + config.W10_CONSENSUS_BONUS_WEIGHT * consensus_bonus
        + config.W11_CLUSTER_QUALITY_BONUS_WEIGHT * cluster_quality_bonus
        + config.W11_CLUSTER_DIVERSITY_BONUS_WEIGHT * cluster_diversity_bonus
        + config.W12_PCA_VARIANCE_BONUS_WEIGHT * pca_variance_bonus
        + config.W13_RL_CREDIT_BONUS_WEIGHT * rl_credit_bonus
        + config.W13_RL_RECENT_SUCCESS_WEIGHT * success_bonus
        + config.W13_RL_EXPLORATION_BONUS_WEIGHT * exploration_bonus
        - config.W12_PCA_REDUNDANCY_PENALTY_WEIGHT * pca_redundancy_penalty
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
        "cluster_target_label": int(cluster_info.get("best_cluster", {}).get("label", -1)) if cluster_info.get("enabled") and "best_cluster" in cluster_info else -1,
        "pca_components": int(cluster_info.get("n_components", 0) or 0),
        "pca_explained_variance": float(cluster_info.get("explained_variance", 0.0) or 0.0),
        "pca_redundancy": float(cluster_info.get("redundancy", 0.0) or 0.0),
        "pca_variance_bonus_at_choice": float(pca_variance_bonus[idx]),
        "pca_redundancy_penalty_at_choice": float(pca_redundancy_penalty[idx]),
        "rl_feedback_enabled": bool(cluster_info.get("rl_feedback", False)),
        "rl_epsilon": float(cluster_info.get("epsilon", 0.0) or 0.0),
        "rl_recent_reward": float(cluster_info.get("recent_reward", 0.0) or 0.0),
        "rl_reward_rate": float(cluster_info.get("reward_rate", 0.0) or 0.0),
        "rl_credit_bonus_at_choice": float(rl_credit_bonus[idx]),
        "rl_recent_success_bonus_at_choice": float(success_bonus[idx]),
        "rl_exploration_bonus_at_choice": float(exploration_bonus[idx]),
        "fast_gp_mode": bool(dim >= config.W9_FAST_GP_DIM_THRESHOLD and len(y) >= config.W9_FAST_GP_POINT_THRESHOLD),
    }
