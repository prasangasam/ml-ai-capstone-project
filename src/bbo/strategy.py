from __future__ import annotations
from typing import Dict
import numpy as np
from . import config

def decide_mode_maximise(y_last: float, y_hist: np.ndarray, tol_frac: float = config.EXPLOIT_TOL_FRAC_OF_RANGE) -> str:
    y_hist = np.asarray(y_hist, float).reshape(-1)
    best = float(np.max(y_hist))
    worst = float(np.min(y_hist))
    span = max(best - worst, 1e-12)
    dist_frac = (best - float(y_last)) / span
    return "exploit" if dist_frac <= tol_frac else "explore"

def tune_params(mode: str, acquisition: str) -> Dict[str, float]:
    a = acquisition.lower().strip()
    if a in ("ei", "pi"):
        return {"xi": config.XI_EXPLOIT if mode == "exploit" else config.XI_EXPLORE}
    if a == "ucb":
        return {"beta": config.BETA_EXPLOIT if mode == "exploit" else config.BETA_EXPLORE}
    raise ValueError("ACQUISITION must be one of: ei, pi, ucb")
