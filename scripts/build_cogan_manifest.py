"""Inventory the historical seven-task BIDS subset. Explicitly review task/version discovery before using for a wider or newer corpus."""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
from dataclasses import asdict, dataclass
from cogan_preprocessing.corpus_ids import cogan_global_id
PLANNED_TASKS: tuple[str, ...] = ('SentenceRep', 'Phoneme_sequencing', 'TIMIT', 'LexicalDecRepDelay', 'Uniqueness_point', 'Neighborhood_Sternberg', 'GlobalLocal')
_IEEG_RE = re.compile('^(?P<subject>sub-D\\d+[A-Z]?)_task-(?P<ftask>[A-Za-z0-9]+)(?:_acq-(?P<acq>[A-Za-z0-9]+))?(?:_run-(?P<run>[A-Za-z0-9]+))?_ieeg\\.edf$')
_SUBJECT_RE = re.compile('sub-D0*(?P<dnum>\\d+)(?P<implant>[A-Z]?)$')

@dataclass(frozen=True)
class ManifestRow:
    subject_bids: str
    d_num: int
    implant: str
    global_subject_id: int
    task: str
    acq: str
    run: str
    trial_id: int
    edf_path: str
    json_path: str
    channels_tsv_path: str
    native_sfreq: float
    power_line_hz: float
    duration_s: float

def parse_subject(subject_bids: str) -> tuple[int, str]:
    """``sub-D0019`` → ``(19, "")``; ``sub-D0107A`` → ``(107, "A")``."""
    m = _SUBJECT_RE.fullmatch(subject_bids)
    if not m:
        raise ValueError(f'unrecognised Cogan BIDS subject {subject_bids!r}')
    return (int(m.group('dnum')), m.group('implant'))

def find_bids_roots(data_root: str) -> dict[str, str]:
    """``{task: <data_root>/BIDS-<ver>_<task>/BIDS}`` for each planned task present."""
    roots: dict[str, str] = {}
    for task in PLANNED_TASKS:
        hits = sorted(glob.glob(os.path.join(data_root, f'BIDS-*_{task}', 'BIDS')))
        if hits:
            roots[task] = hits[0]
    return roots

def _channels_tsv_for(ieeg_dir: str, subject: str, ftask: str) -> str:
    """Per-subject-per-task ``channels.tsv`` path if it exists, else ``""``.

    BIDS names it without acq/run (``sub-D0019_task-PhonemeSequence_channels.tsv``),
    shared across that subject's runs of the task.
    """
    cand = os.path.join(ieeg_dir, f'{subject}_task-{ftask}_channels.tsv')
    return cand if os.path.exists(cand) else ''

def enumerate_runs(bids_roots: dict[str, str]) -> list[ManifestRow]:
    """Walk each task tree → one un-trial-id'd ``ManifestRow`` per ``_ieeg.edf``.

    Runs whose ``_ieeg.json`` sidecar is absent (incomplete/practice EDFs — seen
    on TIMIT D38/D39/D54) are SKIPPED with a warning: without the sidecar we have
    no native rate/duration, and those same runs also lack ``channels.tsv``.
    """
    rows: list[ManifestRow] = []
    skipped: list[str] = []
    for task, root in bids_roots.items():
        for edf in sorted(glob.glob(os.path.join(root, 'sub-D*', 'ieeg', '*_ieeg.edf'))):
            fname = os.path.basename(edf)
            m = _IEEG_RE.match(fname)
            if not m:
                skipped.append(fname)
                continue
            subject = m.group('subject')
            ftask = m.group('ftask')
            acq = m.group('acq') or ''
            run = m.group('run') or ''
            ieeg_dir = os.path.dirname(edf)
            json_path = edf[:-len('_ieeg.edf')] + '_ieeg.json'
            if not os.path.exists(json_path):
                skipped.append(fname)
                continue
            with open(json_path) as fh:
                meta = json.load(fh)
            d_num, implant = parse_subject(subject)
            rows.append(ManifestRow(subject_bids=subject, d_num=d_num, implant=implant, global_subject_id=cogan_global_id(d_num, implant), task=task, acq=acq, run=run, trial_id=-1, edf_path=edf, json_path=json_path, channels_tsv_path=_channels_tsv_for(ieeg_dir, subject, ftask), native_sfreq=float(meta['SamplingFrequency']), power_line_hz=float(meta.get('PowerLineFrequency', 60.0)), duration_s=float(meta['RecordingDuration'])))
    if skipped:
        print(f'[enumerate_runs] skipped {len(skipped)} runs w/o json/unparseable: {skipped[:8]}')
    return rows

def assign_trial_ids(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Per BIDS-subject, enumerate runs 0..N in sorted ``(task, acq, run)`` order.

    A subject appears across multiple tasks; ``trial_id`` must be unique per
    subject (it pairs with ``global_subject_id`` in the cache key), so it spans
    tasks, not per-task.
    """
    by_subject: dict[str, list[ManifestRow]] = {}
    for r in rows:
        by_subject.setdefault(r.subject_bids, []).append(r)
    out: list[ManifestRow] = []
    for subject in sorted(by_subject):
        ordered = sorted(by_subject[subject], key=lambda r: (r.task, r.acq, r.run))
        for tid, r in enumerate(ordered):
            out.append(ManifestRow(**{**asdict(r), 'trial_id': tid}))
    return out

def _assert_ids_disjoint(rows: list[ManifestRow]) -> None:
    """One ``global_subject_id`` ↔ one distinct BIDS subject.

    A re-implant (``sub-D0107A``) collapses to the same ``d_num`` as a bare
    ``D0107`` / ``D0107B`` under ``cogan_global_id`` — distinct brains, colliding
    cache-key ids. If the corpus ever grows such a pair, fail LOUD here rather
    than let two subjects silently share a key namespace.
    """
    owner: dict[int, str] = {}
    for r in rows:
        prev = owner.setdefault(r.global_subject_id, r.subject_bids)
        if prev != r.subject_bids:
            raise ValueError(f'global_subject_id {r.global_subject_id} shared by distinct subjects {prev!r} and {r.subject_bids!r} — re-implant/d_num collision; extend cogan_global_id to fold the implant suffix')

def build(data_root: str) -> list[ManifestRow]:
    rows = assign_trial_ids(enumerate_runs(find_bids_roots(data_root)))
    _assert_ids_disjoint(rows)
    return rows

def write_csv(rows: list[ManifestRow], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = list(ManifestRow.__dataclass_fields__)
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = build(args.data_root)
    write_csv(rows, args.out)
    n_sub = len({r.subject_bids for r in rows})
    n_missing = sum((1 for r in rows if not r.channels_tsv_path))
    print(f'wrote {len(rows)} runs / {n_sub} BIDS subjects → {args.out} ({n_missing} runs missing channels.tsv)')
if __name__ == '__main__':
    main()
