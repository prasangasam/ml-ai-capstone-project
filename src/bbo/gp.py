from __future__ import annotations
from typing import Any, List, Tuple, Optional
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
import warnings
from . import config

warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


def _restarts(dim: int) -> int:
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
    for j, k in enumerate(_kernel_pool(dim)):
        gp = GaussianProcessRegressor(
            kernel=k,
            alpha=config.NOISE_ALPHA,
            normalize_y=True,
            n_restarts_optimizer=_restarts(dim),
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
) -> np.ndarray:
    strategy = strategy.lower().strip()
    if strategy == "explore":
        return rng.uniform(0.0, 1.0, size=(n_candidates, dim))

    best_x = np.asarray(X[int(np.argmax(y))], dtype=float)

    if strategy == "refine":
        n_local = max(1, int(config.W8_LOCAL_CANDIDATE_RATIO * n_candidates))
        n_global = max(1, n_candidates - n_local)
        local_scale = config.W8_REFINEMENT_SCALE_LOW_D if dim <= 4 else config.W8_REFINEMENT_SCALE_HIGH_D
        local = np.clip(rng.normal(loc=best_x, scale=local_scale, size=(n_local, dim)), 0.0, 0.999999)
        global_ = rng.uniform(0.0, 1.0, size=(n_global, dim))
        return np.vstack([local, global_])

    # Week 8: BO uses a mixed local/global candidate set instead of fully global search.
    n_local = max(1, int(0.55 * n_candidates))
    n_global = max(1, n_candidates - n_local)
    local_scale = config.W8_BO_LOCAL_SCALE_LOW_D if dim <= 4 else config.W8_BO_LOCAL_SCALE_HIGH_D
    local = np.clip(rng.normal(loc=best_x, scale=local_scale, size=(n_local, dim)), 0.0, 0.999999)
    global_ = rng.uniform(0.0, 1.0, size=(n_global, dim))
    return np.vstack([local, global_])


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
):
    rng = np.random.default_rng(seed)
    dim = X.shape[1]
    gp, best_lml, lml_details = fit_best_gp_by_lml(X, y, dim=dim, seed=seed)

    Xcand = _build_candidates(rng, X=X, y=y, dim=dim, n_candidates=n_candidates, strategy=strategy)
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

    sigma_bonus = config.W8_SIGMA_BOOST * float(instability) * sigma
    boundary_pen = _boundary_penalty(Xcand)
    repeat_pen = _repeat_penalty(Xcand, X)
    score = raw_score + sigma_bonus - config.W8_BOUNDARY_PENALTY_WEIGHT * boundary_pen - config.W8_REPEAT_PENALTY_WEIGHT * repeat_pen

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
        "raw_score_at_choice": float(raw_score[idx]),
        "adjusted_score_at_choice": float(score[idx]),
        "sigma_bonus_at_choice": float(sigma_bonus[idx]),
        "boundary_penalty_at_choice": float(boundary_pen[idx]),
        "repeat_penalty_at_choice": float(repeat_pen[idx]),
    }
