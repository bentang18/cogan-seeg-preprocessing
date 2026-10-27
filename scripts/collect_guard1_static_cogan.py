"""Collect per-run static contact exclusions with manifest coverage checks."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from cogan_preprocessing import guard1
_REQUIRED_KEYS = ('subject', 'trial', 'labels', 'clip_frac', 'rmad', 'static')

def load_sessions(detections_dir: Path) -> list[dict]:
    files = sorted(detections_dir.glob('cogan*.json'))
    if not files:
        raise SystemExit(f'no cogan*.json signatures under {detections_dir}')
    sessions: list[dict] = []
    for f in files:
        rec = json.loads(f.read_text())
        for key in _REQUIRED_KEYS:
            if key not in rec:
                raise SystemExit(f'{f} missing required key {key!r}')
        sessions.append(rec)
    return sessions

def manifest_run_keys(manifest_path: Path) -> set[tuple[int, int]]:
    """Expected (global_subject_id, trial_id) keys — one per manifest run."""
    with open(manifest_path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    return {(int(r['global_subject_id']), int(r['trial_id'])) for r in rows}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detections-dir', required=True, help='dir of cogan{sid}_t{tid}.json signatures from the precompute')
    ap.add_argument('--manifest', default=None, help='cogan_manifest.csv — if given, enforce full run coverage')
    ap.add_argument('--out', required=True, help='output JSON static-map path')
    args = ap.parse_args()
    sessions = load_sessions(Path(args.detections_dir))
    print(f'[load] {len(sessions)} per-run signatures')
    per_run = guard1.per_session_static_drops(sessions)
    if args.manifest:
        expected = manifest_run_keys(Path(args.manifest))
        present = set(per_run)
        extra = sorted(present - expected)
        missing = sorted(expected - present)
        if extra:
            print(f'[warn] {len(extra)} signature(s) not in manifest: {extra[:10]}' + (' ...' if len(extra) > 10 else ''))
        if missing:
            raise SystemExit(f'[FAIL] {len(missing)} manifest run(s) have NO signature — cannot emit a per-run map without every run: {missing[:10]}' + (' ...' if len(missing) > 10 else ''))
        print(f'[coverage] all {len(expected)} manifest runs present')
    n_drop = sum((1 for v in per_run.values() if v))
    n_contacts = sum((len(v) for v in per_run.values()))
    print(f'[static] {n_drop}/{len(per_run)} runs drop >=1 contact ({n_contacts} contacts total)')
    by_key = {(int(s['subject']), int(s['trial'])): s for s in sessions}
    for key in sorted(per_run):
        v = per_run[key]
        if v:
            rec = by_key[key]
            print(f"[static]   cogan{key[0]}_t{key[1]}: spike={rec.get('spike')} noisy={rec.get('noisy')} dead={rec.get('dead')}")
    report = {'n_runs': len(sessions), 'thresholds': {'spike_clip_frac': guard1.SPIKE_CLIP_FRAC, 'noisy_z': guard1.NOISY_Z, 'dead_ratio': guard1.DEAD_RATIO}, 'n_runs_with_drop': n_drop, 'n_contacts_dropped': n_contacts, 'per_run': {f'{s}_{t}': per_run[s, t] for s, t in sorted(per_run)}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f'[write] {out_path}')
if __name__ == '__main__':
    main()
