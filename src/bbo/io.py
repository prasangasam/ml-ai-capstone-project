from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List
import json
import numpy as np
from . import config

def fmt_query(x: np.ndarray, decimals: int = config.DECIMALS) -> str:
    x = np.asarray(x, float).reshape(-1)
    x = np.clip(x, 0.0, 0.999999)
    return "-".join([f"{v:.{decimals}f}" for v in x])

def ensure_dirs() -> None:
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

def save_week_snapshot(*, week_k: int, payload: Dict[str, Any]):
    ensure_dirs()
    payload = dict(payload)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    out_path = config.HISTORY_DIR / f"week_{week_k:02d}_to_week_{week_k+1:02d}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

def save_submission_file(*, week_next: int, portal_lines: List[str]):
    ensure_dirs()
    out_path = config.SUBMISSIONS_DIR / f"week_{week_next:02d}_queries.txt"
    out_path.write_text("\n".join(portal_lines) + "\n", encoding="utf-8")
    return out_path
