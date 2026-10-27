"""Read BIDS EDF voltage, preserve channel identity, select neural contacts and resample. Adapted from the existing Cogan loader."""
from __future__ import annotations
import csv
import re
from collections.abc import Iterable
from math import gcd
import numpy as np
from scipy.signal import resample_poly
NEURAL_TYPES: frozenset[str] = frozenset({'ECOG', 'SEEG'})
TARGET_RATE_HZ: float = 2048.0
_CONTACT_RE = re.compile('^(?P<shaft>.*?)(?P<depth>\\d+)$')

def parse_contact(name: str) -> tuple[str, int] | None:
    """``ch_name`` → ``(shaft_label, contact_index)`` or ``None`` if it has no
    trailing contact number.

    A missing contact number is the empirical signature (verified over the whole
    Cogan corpus, 4003 distinct neural-typed names, 2026-07-13) of a physiological
    /scalp reference mistyped as ``ECOG``/``SEEG`` — ``EKGL``, ``EKGR``, ``LEMG``,
    ``REMG``, ``Cz``/``Fz``/``Pz``, or a blank name. Every one of the 3992 real
    depth/grid contacts carries a trailing index; none of the 11 refs do. It is
    also a HARD requirement of the model: within-shaft RoPE positions each contact
    by this index, so a contact-less channel cannot be placed.
    """
    m = _CONTACT_RE.match(name.strip())
    if not m:
        return None
    return (m.group('shaft'), int(m.group('depth')))

def read_channel_types(channels_tsv_path: str) -> dict[str, str]:
    """BIDS ``channels.tsv`` → ``{channel_name: TYPE}`` (upper-cased type).

    ``utf-8-sig`` strips a leading UTF-8 BOM if present (13 Cogan subjects — e.g.
    sub-D0057 — carry a ``EF BB BF`` BOM + CRLF from a Windows export, which under a
    plain-utf-8 DictReader mangles the first header into ``\\ufeffname`` and KeyErrors
    on ``row['name']``); it behaves as utf-8 for the BOM-free majority. csv handles
    the CRLF itself under ``newline=''``.
    """
    types: dict[str, str] = {}
    with open(channels_tsv_path, newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            types[row['name']] = (row.get('type') or '').strip().upper()
    return types

def select_neural(ch_names: list[str], data: np.ndarray, ch_type_by_name: dict[str, str], extra_bad: Iterable[str]=()) -> tuple[np.ndarray, list[str]]:
    """Keep neural, non-bad DEPTH/GRID channels in the given (voltage) order.

    ``data`` is ``(n_ch, n_samples)`` row-aligned with ``ch_names``. A channel is
    kept iff its ``channels.tsv`` type is in ``NEURAL_TYPES``, it is not in
    ``extra_bad``, AND it parses as a contact (has a trailing index). The last
    clause drops physio/scalp references mistyped as ``ECOG``/``SEEG``
    (``EKGL``/``LEMG``/``Cz``…) — the type column alone doesn't catch them. A
    channel with no ``channels.tsv`` type is treated as non-neural (dropped) —
    fail-safe, never fail-open.
    """
    if data.shape[0] != len(ch_names):
        raise ValueError(f'data rows {data.shape[0]} != n ch_names {len(ch_names)}')
    bad = set(extra_bad)
    keep_idx = [i for i, nm in enumerate(ch_names) if ch_type_by_name.get(nm, '') in NEURAL_TYPES and nm not in bad and (parse_contact(nm) is not None)]
    if not keep_idx:
        raise ValueError(f'no neural channels survived selection (types seen: {sorted(set(ch_type_by_name.values()))}, n extra_bad={len(bad)})')
    kept_names = [ch_names[i] for i in keep_idx]
    return (np.ascontiguousarray(data[keep_idx], dtype=np.float32), kept_names)

def resample_to(data: np.ndarray, native_rate: float, target_rate: float=TARGET_RATE_HZ) -> np.ndarray:
    """Polyphase-resample ``(n_ch, n_samples)`` from ``native_rate`` → ``target_rate``.

    Identity when the rates match. Uses the reduced integer ratio (e.g. 2000→2048
    = up 128 / down 125; 1024→2048 = up 2 / down 1), matching ``bt_load_raw``.
    """
    nr, tr = (int(round(native_rate)), int(round(target_rate)))
    if nr == tr:
        return np.ascontiguousarray(data, dtype=np.float32)
    if nr <= 0:
        raise ValueError(f'non-positive native_rate {native_rate}')
    g = gcd(nr, tr)
    up, down = (tr // g, nr // g)
    out = resample_poly(data, up, down, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)

def cogan_load_raw(edf_path: str, channels_tsv_path: str, *, extra_bad: Iterable[str]=(), target_rate: float=TARGET_RATE_HZ) -> tuple[np.ndarray, list[str], float]:
    """``(data, ch_names, sfreq)`` for one Cogan BIDS run, v3-ready.

    Reads the EDF via ``mne`` (header gives native ``sfreq``), keeps neural
    non-bad channels in EDF/voltage order, resamples to ``target_rate``.
    """
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    types = read_channel_types(channels_tsv_path)
    data, ch_names = select_neural(list(raw.ch_names), np.asarray(raw.get_data()), types, extra_bad)
    data = resample_to(data, float(raw.info['sfreq']), target_rate)
    return (data, ch_names, float(target_rate))
