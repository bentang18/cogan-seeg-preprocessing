"""Unchanged existing bad-span decision arithmetic, isolated from the legacy CLI."""
from __future__ import annotations
import numpy as np
CLIP_S: float = 5.0
Q_PCT: float = 99.0
FRAC_HOT: float = 0.05
FRAC_FLAT: float = 0.05
N_FLOOR: int = 3
HOT_MULT: float = 4.0
CAT_MULT: float = 8.0
FLAT_STD: float = 0.05
ABS_FLOOR_MAD: float = 200.0
HOT_MULT_BY_BAND: dict[str, float] = {}
CAT_MULT_BY_BAND: dict[str, float] = {'lfs': 6.0, 'hga': 6.0, 'v3hga': 6.0}
SIGMA_FLOOR: float = 1e-06

def _merge_bad_windows(bad_idx: list[int], clip_s: float, total_s: float) -> list[tuple[float, float]]:
    """Merge a sorted list of bad window INDICES into neural-second spans. Window
    ``i`` covers ``[i*clip_s, (i+1)*clip_s)``; the final partial window is capped at
    the session duration. Contiguous bad windows fuse into one span."""
    spans: list[tuple[float, float]] = []
    for i in sorted(bad_idx):
        lo = i * clip_s
        hi = min((i + 1) * clip_s, total_s)
        if spans and lo <= spans[-1][1] + 1e-09:
            spans[-1] = (spans[-1][0], hi)
        else:
            spans.append((lo, hi))
    return spans

def _decide_bad_windows(ewm_by_band: dict[str, np.ndarray], n_flat: np.ndarray, n_elec: int, *, q_pct: float=Q_PCT, frac_hot: float=FRAC_HOT, frac_flat: float=FRAC_FLAT, n_floor: int=N_FLOOR, hot_mult: float=HOT_MULT, cat_mult: float=CAT_MULT, abs_floor_mad: float=ABS_FLOOR_MAD, hot_mult_by_band: dict[str, float] | None=None, cat_mult_by_band: dict[str, float] | None=None) -> tuple[list[int], dict]:
    """Pure PER-BAND bad-window decision (#231) — the testable core, no I/O.

    ``ewm_by_band[name]`` is ``(n_elec, n_windows)``: each electrode's per-window
    |z|-max WITHIN that band. ``n_flat`` is ``(n_windows,)``: the count of
    slow-band-flat electrodes per window. Each band self-calibrates its own scale
    ``q_band`` (P99 of its own (elec, window) |z|-max) and fences ride above it; a
    window is BAD iff ANY band fires hot (common-mode), cat (single-cell relative to
    q), or abs (single-cell above the q-INDEPENDENT ``abs_floor_mad`` backstop), OR
    the slow band fires flat (dropout). Returns ``(sorted bad_idx, diagnostics)``."""
    hot_mult_by_band = hot_mult_by_band or {}
    cat_mult_by_band = cat_mult_by_band or {}
    n_windows = int(n_flat.shape[0])
    hot_count_thresh = max(n_floor, int(np.ceil(frac_hot * n_elec)))
    flat_count_thresh = max(n_floor, int(np.ceil(frac_flat * n_elec)))
    hot_bad = np.zeros(n_windows, bool)
    cat_bad = np.zeros(n_windows, bool)
    abs_bad = np.zeros(n_windows, bool)
    per_band: dict[str, dict] = {}
    for name, ewm in ewm_by_band.items():
        q = float(np.percentile(ewm, q_pct)) if ewm.size else 0.0
        hm = float(hot_mult_by_band.get(name, hot_mult))
        cm = float(cat_mult_by_band.get(name, cat_mult))
        hot_fence, cat_fence = (hm * q, cm * q)
        win_cell_max = ewm.max(axis=0) if ewm.shape[0] else np.zeros(n_windows, np.float32)
        n_hot = (ewm > hot_fence).sum(axis=0).astype(np.int32)
        hb = n_hot >= hot_count_thresh
        cb = win_cell_max > cat_fence
        ab = win_cell_max > abs_floor_mad
        hot_bad |= hb
        cat_bad |= cb
        abs_bad |= ab
        per_band[name] = {'q': q, 'hot_mult': hm, 'cat_mult': cm, 'hot_fence': float(hot_fence), 'cat_fence': float(cat_fence), 'n_hot_windows': int(hb.sum()), 'n_cat_windows': int(cb.sum()), 'n_abs_windows': int(ab.sum()), 'max_n_hot': int(n_hot.max()) if n_windows else 0, 'max_cell_max': float(win_cell_max.max()) if n_windows else 0.0}
    flat_bad = n_flat >= flat_count_thresh
    bad = hot_bad | cat_bad | abs_bad | flat_bad
    bad_idx = [int(w) for w in np.nonzero(bad)[0]]
    diag = {'q_pct': q_pct, 'frac_hot': frac_hot, 'frac_flat': frac_flat, 'n_floor': n_floor, 'flat_std': FLAT_STD, 'abs_floor_mad': abs_floor_mad, 'hot_count_thresh': int(hot_count_thresh), 'flat_count_thresh': int(flat_count_thresh), 'per_band': per_band, 'n_hot_windows': int(hot_bad.sum()), 'n_cat_windows': int(cat_bad.sum()), 'n_abs_windows': int(abs_bad.sum()), 'n_flat_windows': int(flat_bad.sum()), 'max_n_flat': int(n_flat.max()) if n_windows else 0}
    return (bad_idx, diag)
