"""Scan compatible pre-existing three-band MAPA caches for bad time spans. This tool does not generate frontend caches."""
from __future__ import annotations
import argparse
import csv
import json
import os
import numpy as np
from cogan_preprocessing.cache_index import index_band_cache, parse_key_session
from cogan_preprocessing.guard2 import CAT_MULT_BY_BAND, FLAT_STD, HOT_MULT_BY_BAND, SIGMA_FLOOR, _decide_bad_windows, _merge_bad_windows
BAND_NAMES: tuple[str, ...] = ('v3slow', 'v3mid', 'hga')
FLAT_BAND: str = 'v3slow'
FPS: float = 32.0
DETECT_WINDOW_S: float = 1.0

def compute_ewm_from_cache(band_mags: list[np.ndarray], band_stats: list[tuple[np.ndarray, np.ndarray]], band_names: tuple[str, ...], *, flat_band: str, clip_s: float, fps: float, sigma_floor: float=SIGMA_FLOOR) -> tuple[dict[str, np.ndarray], np.ndarray, int, float, int]:
    """Cached |STFT| magnitudes → the detector's per-band ``ewm`` grid + ``n_flat``.

    ``band_mags[b]`` is ``(C, F_b, T)`` raw magnitude (the ``.npy``); ``band_stats[b]``
    is the band's frozen ``(median, sigma)`` (each ``(C, F_b)`` or ``(C, F_b, 1)``).
    Applies the SAME robust-z the encoder reads — median/σ per (channel, freq-bin)
    — but WITHOUT the read-time winsor clamp (guard-2 detects pre-winsor). Returns
    ``(ewm_by_band, n_flat, n_elec, total_s, n_windows)`` in the exact shapes
    ``_decide_bad_windows`` consumes, so the detector is reused untouched."""
    T = int(band_mags[0].shape[2])
    for name, mag in zip(band_names, band_mags):
        if int(mag.shape[2]) != T:
            raise ValueError(f"band '{name}' has {mag.shape[2]} frames, expected {T} (band caches must share the 32 Hz time axis)")
    n_elec = int(band_mags[0].shape[0])
    total_s = T / float(fps)
    n_windows = int(np.ceil(total_s / clip_s)) if T else 0
    t_start = np.arange(T) / float(fps)
    win = np.clip((t_start / clip_s).astype(np.int64), 0, max(n_windows - 1, 0))
    ii = np.broadcast_to(np.arange(n_elec)[:, None], (n_elec, T))
    jj = np.broadcast_to(win[None, :], (n_elec, T))
    ewm_by_band: dict[str, np.ndarray] = {}
    n_flat = np.zeros(n_windows, np.int32)
    for name, mag, (median, sigma) in zip(band_names, band_mags, band_stats):
        mag = np.asarray(mag, dtype=np.float32)
        med = np.asarray(median, dtype=np.float32)
        sig = np.asarray(sigma, dtype=np.float32)
        if med.ndim == 2:
            med = med[:, :, None]
        if sig.ndim == 2:
            sig = sig[:, :, None]
        z = (mag - med) / np.maximum(sig, sigma_floor)
        per_frame = np.abs(z).max(axis=1)
        ewm = np.zeros((n_elec, n_windows), np.float32)
        np.maximum.at(ewm, (ii, jj), per_frame)
        ewm_by_band[name] = ewm
        if name == flat_band:
            n_bins = z.shape[1]
            fsum = z.sum(axis=1)
            fsq = (z * z).sum(axis=1)
            wsum = np.zeros((n_elec, n_windows), np.float64)
            wsq = np.zeros((n_elec, n_windows), np.float64)
            wcnt = np.zeros(n_windows, np.float64)
            np.add.at(wsum, (ii, jj), fsum)
            np.add.at(wsq, (ii, jj), fsq)
            np.add.at(wcnt, win, float(n_bins))
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = wsum / wcnt
                var = wsq / wcnt - mean * mean
            elec_std = np.sqrt(np.clip(var, 0.0, None))
            elec_std[:, wcnt <= 0] = np.inf
            n_flat = (elec_std < FLAT_STD).sum(axis=0).astype(np.int32)
    return (ewm_by_band, n_flat, n_elec, total_s, n_windows)

def _entry_for(index: dict, subject_id: int, trial_id: int):
    """The unique cache entry for a session, matched on the sidecar ``key`` (the
    same match ``load_v3_sessions`` uses). >1 or 0 is a fail-loud."""
    hits = [e for k, e in index.items() if parse_key_session(k) == (subject_id, trial_id)]
    if len(hits) != 1:
        raise ValueError(f'expected exactly one cache entry for subject {subject_id} trial {trial_id}, found {len(hits)}')
    return hits[0]

def scan_session_cache(subject_id: int, trial_id: int, band_cache_dirs: list[str], *, clip_s: float=DETECT_WINDOW_S, fps: float=FPS) -> dict:
    """Scan one session's baked spec cache → its bad-window sidecar payload.

    ``band_cache_dirs`` are the 3 band roots in v3 concat order (slow, mid, hga);
    each is resolved to its ``band_hop=64`` leaf by ``index_band_cache``."""
    indexes = [index_band_cache(d) for d in band_cache_dirs]
    entries = [_entry_for(bi, subject_id, trial_id) for bi in indexes]
    ch0 = entries[0].ch_names
    for e in entries[1:]:
        if e.ch_names != ch0:
            raise ValueError(f'band caches disagree on channel order for session {subject_id}/{trial_id}: {e.ch_names[:3]}... vs {ch0[:3]}...')
    total_frames = entries[0].total_frames
    for e in entries[1:]:
        if e.total_frames != total_frames:
            raise ValueError(f'band caches disagree on total_frames for session {subject_id}/{trial_id}: {e.total_frames} vs {total_frames}')
    for e in entries:
        sr = int(e.sample_rate)
        fps_i = int(round(fps))
        if sr != fps_i and (sr <= 0 or sr % fps_i != 0):
            raise ValueError(f'cache sample_rate {sr} is not on the {fps_i} Hz frame clock (expected {fps_i} or an integer multiple) for session {subject_id}/{trial_id}')
    band_mags: list[np.ndarray] = []
    band_stats: list[tuple[np.ndarray, np.ndarray]] = []
    for e in entries:
        mm = np.load(e.npy_path, mmap_mode='r')
        band_mags.append(np.asarray(mm, dtype=np.float32))
        del mm
        st = np.load(e.stats_path)
        band_stats.append((st['median'], st['sigma']))
    ewm_by_band, n_flat, n_elec, total_s, n_windows = compute_ewm_from_cache(band_mags, band_stats, BAND_NAMES, flat_band=FLAT_BAND, clip_s=clip_s, fps=fps)
    bad_idx, decision = _decide_bad_windows(ewm_by_band, n_flat, n_elec, hot_mult_by_band=HOT_MULT_BY_BAND, cat_mult_by_band=CAT_MULT_BY_BAND)
    bad_windows_s = _merge_bad_windows(bad_idx, clip_s, total_s)
    tag = f'cogan{int(subject_id)}_t{int(trial_id)}'
    return {'session': tag, 'subject_id': int(subject_id), 'trial_id': int(trial_id), 'bad_windows_s': [[float(lo), float(hi)] for lo, hi in bad_windows_s], 'rule': {'clip_s': clip_s, **decision}, 'n_elec': n_elec, 'duration_s': float(total_s), 'n_windows': int(n_windows), 'n_bad_windows': int(len(bad_idx)), 'frac_bad': float(len(bad_idx) / n_windows) if n_windows else 0.0, 'max_n_flat': int(n_flat.max()) if n_windows else 0, 'n_hot_windows': decision['n_hot_windows'], 'n_cat_windows': decision['n_cat_windows'], 'n_abs_windows': decision['n_abs_windows'], 'n_flat_windows': decision['n_flat_windows']}

def _band_dirs(spec_cache_dir: str) -> list[str]:
    """The 3 band roots under a v3 spec-cache dir (dispatch_v14 ``band_{name}``)."""
    return [os.path.join(spec_cache_dir, f'band_{b}') for b in BAND_NAMES]

def _manifest_sessions(manifest_path: str) -> list[tuple[int, int]]:
    """Manifest CSV → ordered ``(global_subject_id, trial_id)`` list (array index)."""
    with open(manifest_path, newline='') as fh:
        reader = csv.DictReader(fh)
        return [(int(r['global_subject_id']), int(r['trial_id'])) for r in reader]

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', required=True, help='sidecar output directory (= v3 bad_window_dir / span_dir)')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--spec-cache-dir', help='v3 spec-cache dir holding band_{v3slow,v3mid,hga}')
    src.add_argument('--band-dir', nargs=3, metavar=('SLOW', 'MID', 'HGA'), help='the 3 band cache roots explicitly (v3 concat order)')
    ap.add_argument('--session', nargs=2, type=int, metavar=('SUBJECT', 'TRIAL'), help='scan exactly this one (global_subject_id, trial_id)')
    ap.add_argument('--manifest', help='cogan manifest CSV; SLURM_ARRAY_TASK_ID indexes its rows')
    ap.add_argument('--detect-window', type=float, default=DETECT_WINDOW_S, help=f'detection-window seconds (default {DETECT_WINDOW_S}; tiling stride AND merged-span resolution)')
    ap.add_argument('--fps', type=float, default=FPS, help=f'frame rate (default {FPS})')
    args = ap.parse_args()
    band_dirs = _band_dirs(args.spec_cache_dir) if args.spec_cache_dir else list(args.band_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.session is not None:
        todo = [(args.session[0], args.session[1])]
    elif args.manifest is not None:
        sessions = _manifest_sessions(args.manifest)
        task = os.environ.get('SLURM_ARRAY_TASK_ID')
        todo = [sessions[int(task)]] if task is not None else sessions
    else:
        ap.error('provide --session S T or --manifest CSV')
    for subject_id, trial_id in todo:
        result = scan_session_cache(subject_id, trial_id, band_dirs, clip_s=args.detect_window, fps=args.fps)
        out_path = os.path.join(args.out_dir, f"{result['session']}.json")
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        r = result['rule']
        bands = '  '.join((f"{name}[q={pb['q']:.1f},cell_max={pb['max_cell_max']:.0f},nhot={pb['max_n_hot']}]" for name, pb in r['per_band'].items()))
        print(f"[{result['session']}] {result['duration_s'] / 60:.1f}min  {result['n_windows']} win  bad={result['n_bad_windows']} ({100 * result['frac_bad']:.2f}%)  by_rule[hot={result['n_hot_windows']},cat={result['n_cat_windows']},abs={result['n_abs_windows']},flat={result['n_flat_windows']}]  {bands}  -> {out_path}", flush=True)
if __name__ == '__main__':
    main()
