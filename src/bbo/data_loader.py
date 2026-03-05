from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

@dataclass
class FunctionSeed:
    idx: int
    X: np.ndarray
    y: np.ndarray

def load_initial_from_dir(initial_dir: Path) -> List[FunctionSeed]:
    if not initial_dir.exists():
        raise FileNotFoundError(f"Initial directory not found: {initial_dir}")

    seeds: List[FunctionSeed] = []
    for i in range(1, 9):
        fdir = initial_dir / f"function_{i}"
        xin = fdir / "initial_inputs.npy"
        yout = fdir / "initial_outputs.npy"
        if not xin.exists() or not yout.exists():
            raise FileNotFoundError(f"Missing initial files for function {i}: {xin} / {yout}")
        X = np.load(xin)
        y = np.load(yout)
        seeds.append(FunctionSeed(i, np.asarray(X, float), np.asarray(y, float).reshape(-1)))
    return seeds

# --- matrix weekly files ---
_PORTAL_TOKEN_RE = re.compile(r"0\.\d{1,6}(?:-0\.\d{1,6})+")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

def _split_row(row: str) -> List[str]:
    return [p for p in re.split(r"[,\s]+", row.strip()) if p]

def _parse_portal_token(tok: str) -> np.ndarray:
    return np.array([float(p) for p in tok.strip().split("-")], dtype=float)

def load_weekly_matrix(weekly_dir: Path) -> Optional[Tuple[List[List[np.ndarray]], List[List[float]]]]:
    in_path = weekly_dir / "inputs.txt"
    out_path = weekly_dir / "outputs.txt"
    if not in_path.exists() or not out_path.exists():
        return None

    in_text = in_path.read_text(encoding="utf-8", errors="ignore")
    out_text = out_path.read_text(encoding="utf-8", errors="ignore")
    
    # Handle line wrapping by joining continuation lines
    in_lines = in_text.splitlines()
    out_lines = out_text.splitlines()
    
    # Fix wrapped lines in inputs (join lines that don't start with '[')
    in_rows = []
    current_row = ""
    for line in in_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('[') and current_row:
            # Start of new row, save previous
            in_rows.append(current_row)
            current_row = line
        elif line.startswith('['):
            # First row
            current_row = line
        else:
            # Continuation line
            current_row += line.lstrip()
    if current_row:
        in_rows.append(current_row)
    
    # Fix wrapped lines in outputs (similar logic)
    out_rows = []
    current_row = ""
    for line in out_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('[') and current_row:
            # Start of new row, save previous
            out_rows.append(current_row)
            current_row = line
        elif line.startswith('['):
            # First row
            current_row = line
        else:
            # Continuation line
            current_row += line.lstrip()
    if current_row:
        out_rows.append(current_row)
    
    if len(in_rows) != len(out_rows):
        raise ValueError(f"inputs.txt and outputs.txt must have same number of rows. Found {len(in_rows)} input rows and {len(out_rows)} output rows.")

    # Write back the cleaned files to prevent future wrapping issues
    in_path.write_text("\n".join(in_rows), encoding="utf-8")
    out_path.write_text("\n".join(out_rows), encoding="utf-8")

    weekly_inputs_all: List[List[np.ndarray]] = []
    weekly_outputs_all: List[List[float]] = []

    for r_in, r_out in zip(in_rows, out_rows):
        # Try portal format first
        toks = _split_row(r_in)
        if toks and toks[0].isdigit():
            toks = toks[1:]
        portal_tokens = [t for t in toks if _PORTAL_TOKEN_RE.fullmatch(t)]
        if len(portal_tokens) != 8:
            portal_tokens = _PORTAL_TOKEN_RE.findall(r_in)
        
        if len(portal_tokens) == 8:
            weekly_inputs_all.append([_parse_portal_token(t) for t in portal_tokens])
        else:
            # Try Python array format (like parse_week_inputs_text does)
            try:
                safe_globals = {"__builtins__": {}, "array": np.array, "np": np}
                obj = eval(r_in, safe_globals, {})
                if isinstance(obj, list) and len(obj) >= 8:
                    weekly_inputs_all.append([np.asarray(x, float).reshape(-1) for x in obj[:8]])
                else:
                    raise ValueError(f"Could not parse 8 inputs from Python array: {r_in}")
            except Exception as e:
                raise ValueError(f"Could not parse inputs from: {r_in}. Error: {e}")

        otoks = _split_row(r_out)
        if otoks and otoks[0].isdigit():
            otoks = otoks[1:]
        floats: List[float] = []
        for t in otoks:
            if _FLOAT_RE.fullmatch(t):
                floats.append(float(t))
        if len(floats) < 8:
            floats = [float(x) for x in _FLOAT_RE.findall(r_out)]
        
        if len(floats) < 8:
            # Try Python array format
            try:
                safe_globals = {"__builtins__": {}, "array": np.array, "np": np, "float64": np.float64}
                obj = eval(r_out, safe_globals, {})
                if isinstance(obj, list) and len(obj) >= 8:
                    floats = [float(x) for x in obj[:8]]
                else:
                    raise ValueError(f"Could not parse 8 outputs from Python array: {r_out}")
            except Exception as e:
                raise ValueError(f"Could not parse outputs from: {r_out}. Error: {e}")
        
        weekly_outputs_all.append(floats[:8])

    return weekly_inputs_all, weekly_outputs_all

# --- legacy per-week files ---
_PORTAL_LINE_RE = re.compile(r"^\s*(0\.\d{1,6}(?:-0\.\d{1,6})+)\s*$")
_FUNCTION_VAL_RE = re.compile(r"Function\s*(\d+)\s*:\s*([-+0-9.eE]+)")

def parse_week_inputs_text(text: str) -> List[np.ndarray]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    portal_hits = [ln for ln in lines if _PORTAL_LINE_RE.match(ln)]
    if len(portal_hits) >= 8:
        return [_parse_portal_token(ln) for ln in portal_hits[:8]]
    safe_globals = {"__builtins__": {}, "array": np.array, "np": np}
    obj = eval(text, safe_globals, {})
    if isinstance(obj, list) and len(obj) >= 8:
        return [np.asarray(x, float).reshape(-1) for x in obj[:8]]
    raise ValueError("Could not parse week inputs (legacy).")

def parse_week_outputs_text(text: str) -> List[float]:
    matches = _FUNCTION_VAL_RE.findall(text)
    if matches:
        vals: Dict[int, float] = {}
        for k, v in matches:
            vals[int(k)] = float(v)
        if all(i in vals for i in range(1, 9)):
            return [vals[i] for i in range(1, 9)]
    floats: List[float] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            floats.append(float(ln))
        except ValueError:
            continue
    if len(floats) >= 8:
        return floats[:8]
    raise ValueError("Could not parse week outputs (legacy).")

def load_weekly_per_week_files(weekly_dir: Path) -> Tuple[List[List[np.ndarray]], List[List[float]]]:
    input_files = sorted(weekly_dir.glob("week_*_inputs.txt"))
    if not input_files:
        raise FileNotFoundError("No weekly input files found (week_XX_inputs.txt).")
    weekly_inputs_all: List[List[np.ndarray]] = []
    weekly_outputs_all: List[List[float]] = []
    for in_path in input_files:
        m = re.search(r"week_(\d+)_inputs\.txt$", in_path.name)
        if not m:
            continue
        w = int(m.group(1))
        out_path = weekly_dir / f"week_{w:02d}_outputs.txt"
        if not out_path.exists():
            raise FileNotFoundError(f"Missing outputs file for week {w:02d}: {out_path.name}")
        weekly_inputs_all.append(parse_week_inputs_text(in_path.read_text(encoding="utf-8", errors="ignore")))
        weekly_outputs_all.append(parse_week_outputs_text(out_path.read_text(encoding="utf-8", errors="ignore")))
    return weekly_inputs_all, weekly_outputs_all

def load_weekly(weekly_dir: Path) -> Tuple[List[List[np.ndarray]], List[List[float]], str]:
    m = load_weekly_matrix(weekly_dir)
    if m is not None:
        return m[0], m[1], "matrix"
    wi, wo = load_weekly_per_week_files(weekly_dir)
    return wi, wo, "per-week-files"
