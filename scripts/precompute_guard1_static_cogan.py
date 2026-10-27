"""Scan one manifest run for static bad contacts before referencing. Inputs stay unchanged; detections go to caller-specified derived storage."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import numpy as np
HPF_HZ: float = 0.5

def read_manifest(manifest_path: str) -> list[dict]:
    """Manifest CSV -> ordered list of per-run dicts (row order = array order).

    Keeps only the fields the scan needs: ``subject_id`` (= ``global_subject_id``),
    ``trial_id``, ``edf_path``, ``channels_tsv_path``, ``power_line_hz``.
    """
    rows: list[dict] = []
    with open(manifest_path, newline='') as fh:
        for r in csv.DictReader(fh):
            rows.append({'subject_id': int(r['global_subject_id']), 'trial_id': int(r['trial_id']), 'edf_path': r['edf_path'], 'channels_tsv_path': r['channels_tsv_path'], 'power_line_hz': float(r['power_line_hz'])})
    return rows

def resolve_rows(manifest: list[dict], row_arg: int | None) -> list[dict]:
    """Rows to scan: ``--row N`` picks one, else ``SLURM_ARRAY_TASK_ID`` indexes the
    manifest, else the whole manifest (serial)."""
    if row_arg is not None:
        return [manifest[row_arg]]
    task = os.environ.get('SLURM_ARRAY_TASK_ID')
    return [manifest[int(task)]] if task is not None else manifest

def scan_run(row: dict) -> dict:
    """Scan one Cogan run and return its signature payload."""
    import mne
    mne.set_log_level('ERROR')
    from cogan_preprocessing import guard1
    from cogan_preprocessing.loader import cogan_load_raw
    data, ch_names, sfreq = cogan_load_raw(row['edf_path'], row['channels_tsv_path'], extra_bad=())
    ch_names = list(ch_names)
    sfreq = float(sfreq)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='seeg', verbose=False)
    raw = mne.io.RawArray(np.asarray(data, dtype=np.float64), info, verbose=False)
    raw.load_data()
    raw.notch_filter(row['power_line_hz'], phase='zero', filter_length='auto', verbose=False)
    raw.filter(HPF_HZ, None, verbose=False)
    filt = np.asarray(raw.get_data(), dtype=np.float64)
    del raw, data
    gc.collect()
    sig = guard1.static_bad_mask(filt, sfreq)
    payload = guard1.session_signature(row['subject_id'], row['trial_id'], ch_names, sig)
    payload['fs'] = sfreq
    payload['n_elec'] = int(filt.shape[0])
    payload['n_samples'] = int(filt.shape[1])
    payload['power_line_hz'] = row['power_line_hz']
    return payload

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True, help='cogan_manifest.csv path')
    ap.add_argument('--out-dir', required=True, help='per-run signature output dir (= collector --detections-dir)')
    ap.add_argument('--row', type=int, default=None, help='scan exactly this one 0-based manifest row (overrides array)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = read_manifest(args.manifest)
    todo = resolve_rows(manifest, args.row)
    for row in todo:
        payload = scan_run(row)
        sid, tid = (payload['subject'], payload['trial'])
        out_path = os.path.join(args.out_dir, f'cogan{sid}_t{tid}.json')
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[cogan{sid}_t{tid}] {payload['n_elec']} elec  {payload['n_samples'] / payload['fs'] / 60:.1f}min  notch={payload['power_line_hz']}Hz  spike={payload['spike']}  noisy={payload['noisy']}  dead={payload['dead']}  -> static={payload['static']}  -> {out_path}", flush=True)
if __name__ == '__main__':
    main()
