"""TDD for the GUARD-1 static detectors. Each test injects a contact with a
known pathology into an otherwise-clean cohort and asserts the right detector
fires, the wrong ones don't (orthogonality), and the clip-router + per-subject
aggregation behave per the Ben-locked spec."""
from __future__ import annotations
import numpy as np
import pytest
from cogan_preprocessing import guard1

def _clean_cohort(n_chan: int, n_clip: int, fs: float, base: float=1.0, seed: int=0) -> np.ndarray:
    """iid Gaussian cohort, ``(n_chan, n_clip·clip_s·fs)``, per-channel std=base."""
    rng = np.random.default_rng(seed)
    t = int(n_clip * guard1.SPIKE_CLIP_S * fs)
    return (rng.standard_normal((n_chan, t)) * base).astype(np.float64)

def test_clean_cohort_fires_nothing() -> None:
    v = _clean_cohort(40, 15, fs=512.0)
    m = guard1.static_bad_mask(v, fs=512.0)
    assert not m['spike'].any(), 'clean gaussian dV/dt must not breach 100σ'
    assert not m['noisy'].any()
    assert not m['dead'].any()
    assert not m['static'].any()

def test_detect_noisy_high_amplitude_and_orthogonal() -> None:
    v = _clean_cohort(40, 15, fs=512.0, seed=1)
    v[7] *= 10.0
    noisy, _ = guard1.detect_noisy(v)
    dead, _ = guard1.detect_dead(v)
    spike, _ = guard1.detect_spike(v, fs=512.0)
    assert noisy[7] and noisy.sum() == 1, 'only the 10x contact is noisy'
    assert not dead[7] and (not spike[7]), 'wide-amplitude ≠ dead/spike (orthogonal)'

def test_detect_dead_attenuated_and_orthogonal() -> None:
    v = _clean_cohort(40, 15, fs=512.0, seed=2)
    v[13] *= 0.1
    dead, ratio = guard1.detect_dead(v)
    noisy, _ = guard1.detect_noisy(v)
    assert dead[13] and dead.sum() == 1
    assert ratio[13] < guard1.DEAD_RATIO
    assert not noisy[13], 'attenuated contact must not also read noisy'

def test_dead_ratio_threshold_is_relative() -> None:
    v = _clean_cohort(40, 15, fs=512.0, seed=3)
    v[5] *= 0.5
    v[6] *= 0.2
    dead, _ = guard1.detect_dead(v)
    assert not dead[5] and dead[6]

def _inject_spike(v: np.ndarray, ch: int, clip_idxs: list[int], fs: float, amp: float=500.0) -> None:
    win = int(round(guard1.SPIKE_CLIP_S * fs))
    for ci in clip_idxs:
        v[ch, ci * win + win // 2] += amp

def test_detect_spike_transient_clips_static_and_orthogonal() -> None:
    fs = 512.0
    v = _clean_cohort(40, 15, fs=fs, seed=4)
    _inject_spike(v, 9, clip_idxs=[2, 7, 11], fs=fs)
    spike, clip_frac = guard1.detect_spike(v, fs=fs)
    noisy, _ = guard1.detect_noisy(v)
    dead, _ = guard1.detect_dead(v)
    assert spike[9] and spike.sum() == 1
    assert clip_frac[9] > guard1.SPIKE_CLIP_FRAC
    assert 0.15 < clip_frac[9] < 0.25
    assert not noisy[9] and (not dead[9]), 'robust spread ⊥ transients'

def test_spike_sparse_below_1pct_router_not_static() -> None:
    fs = 256.0
    v = _clean_cohort(5, 150, fs=fs, seed=5)
    _inject_spike(v, 3, clip_idxs=[40], fs=fs)
    spike, clip_frac = guard1.detect_spike(v, fs=fs)
    assert 0.0 < clip_frac[3] < guard1.SPIKE_CLIP_FRAC
    assert not spike[3], "sub-1% clip-corruption is CLIP's job, not STATIC"

def test_cohort_z_lower_bounded_by_rel_floor() -> None:
    v = np.array([1.0, 1.0, 1.0, 1.0, 1e-06])
    z = guard1.cohort_z(v)
    assert z.min() >= -1.0 / guard1.COHORT_REL_FLOOR - 1e-06
    assert z[-1] == pytest.approx(-1.0 / guard1.COHORT_REL_FLOOR, rel=0.001)

def test_static_is_union_of_three() -> None:
    fs = 512.0
    v = _clean_cohort(40, 15, fs=fs, seed=6)
    v[7] *= 10.0
    v[13] *= 0.1
    _inject_spike(v, 9, [2, 7, 11], fs=fs)
    m = guard1.static_bad_mask(v, fs=fs)
    idx = set(np.flatnonzero(m['static']).tolist())
    assert idx == {7, 9, 13}

def test_aggregate_subject_spike_majority_noisy_dead_any() -> None:
    per_trial = [{'spike': {'A', 'B'}, 'noisy': {'C'}, 'dead': set()}, {'spike': {'B'}, 'noisy': set(), 'dead': {'D'}}, {'spike': set(), 'noisy': set(), 'dead': set()}]
    agg = guard1.aggregate_subject(per_trial)
    assert agg['spike'] == frozenset({'B'})
    assert agg['noisy'] == frozenset({'C'})
    assert agg['dead'] == frozenset({'D'})
    assert agg['static'] == frozenset({'B', 'C', 'D'})

def test_aggregate_subject_empty() -> None:
    agg = guard1.aggregate_subject([])
    assert agg['static'] == frozenset()

def test_classify_from_signature_matches_live_scan() -> None:
    fs = 512.0
    v = _clean_cohort(40, 15, fs=fs, seed=6)
    v[7] *= 10.0
    v[13] *= 0.1
    _inject_spike(v, 9, [2, 7, 11], fs=fs)
    m = guard1.static_bad_mask(v, fs=fs)
    labels = [f'E{i}' for i in range(40)]
    cls = guard1.classify_from_signature(labels, m['clip_frac'], m['rmad'])
    assert cls['spike'] == {'E9'}
    assert cls['noisy'] == {'E7'}
    assert cls['dead'] == {'E13'}

def test_session_signature_roundtrips_through_collector() -> None:
    import json
    fs = 512.0
    v = _clean_cohort(40, 15, fs=fs, seed=6)
    v[7] *= 10.0
    v[13] *= 0.1
    _inject_spike(v, 9, [2, 7, 11], fs=fs)
    sig = guard1.static_bad_mask(v, fs=fs)
    labels = [f'E{i}' for i in range(40)]
    payload = guard1.session_signature(2, 4, labels, sig)
    json.dumps(payload)
    for key in ('subject', 'trial', 'labels', 'clip_frac', 'rmad'):
        assert key in payload
    assert payload['subject'] == 2 and payload['trial'] == 4
    assert payload['spike'] == ['E9']
    assert payload['noisy'] == ['E7']
    assert payload['dead'] == ['E13']
    cls = guard1.classify_from_signature(payload['labels'], np.asarray(payload['clip_frac']), np.asarray(payload['rmad']))
    assert cls['spike'] == set(payload['spike'])
    assert cls['noisy'] == set(payload['noisy'])
    assert cls['dead'] == set(payload['dead'])
    per_subject = guard1.collate_sessions([payload])
    assert per_subject[2]['static'] == frozenset({'E7', 'E9', 'E13'})

def test_classify_from_signature_thresholds_are_locked() -> None:
    labels = ['a', 'b', 'c', 'd']
    rmad = np.array([1.0, 1.0, 1.0, 0.2])
    clip = np.array([0.02, 0.005, 0.0, 0.0])
    cls = guard1.classify_from_signature(labels, clip, rmad)
    assert cls['spike'] == {'a'}
    assert cls['dead'] == {'d'}
    assert cls['noisy'] == set()

def _session(subject: int, trial: int, labels, clip_frac, rmad) -> dict:
    return {'subject': subject, 'trial': trial, 'labels': list(labels), 'clip_frac': list(clip_frac), 'rmad': list(rmad)}

def test_collate_sessions_aggregates_and_classifies_per_subject() -> None:
    labels = ['a', 'b', 'c', 'd']
    rmad = [1.0, 1.0, 1.0, 0.2]
    s1 = _session(2, 0, labels, [0.02, 0.0, 0.0, 0.0], rmad)
    s2 = _session(2, 1, labels, [0.02, 0.0, 0.0, 0.0], rmad)
    out = guard1.collate_sessions([s1, s2])
    assert set(out) == {2}
    sub = out[2]
    assert sub['n_trials'] == 2
    assert sub['dead'] == frozenset({'d'})
    assert sub['spike'] == frozenset({'a'})
    assert sub['static'] == frozenset({'a', 'd'})
    assert sub['chronic'] == frozenset({'a', 'd'})
    assert sub['variable'] == frozenset()

def test_collate_sessions_chronic_vs_variable_split() -> None:
    labels = ['a', 'b', 'c', 'd']
    rmad = [1.0, 1.0, 1.0, 0.2]
    sess = [_session(7, 0, labels, [0.02, 0.0, 0.0, 0.0], rmad), _session(7, 1, labels, [0.0, 0.0, 0.0, 0.0], rmad), _session(7, 2, labels, [0.0, 0.0, 0.0, 0.0], rmad)]
    out = guard1.collate_sessions(sess)[7]
    assert out['n_trials'] == 3
    assert 'a' in out['variable'] and 'a' not in out['chronic']
    assert out['spike'] == frozenset()
    assert out['chronic'] == frozenset({'d'})
    assert out['static'] == frozenset({'d'})

def test_collate_sessions_groups_multiple_subjects() -> None:
    labels = ['a', 'b']
    rmad = [1.0, 0.2]
    sess = [_session(2, 0, labels, [0.0, 0.0], rmad), _session(4, 0, labels, [0.0, 0.0], rmad)]
    out = guard1.collate_sessions(sess)
    assert set(out) == {2, 4}
    assert out[2]['dead'] == frozenset({'b'}) and out[4]['dead'] == frozenset({'b'})

def test_diff_static_drops_added_removed_unchanged() -> None:
    would_be = {2: frozenset({'X', 'Y'}), 9: frozenset({'P2e6'})}
    locked = {2: frozenset({'Y', 'Z'}), 8: frozenset({'Q'})}
    d = guard1.diff_static_drops(would_be, locked)
    assert set(d) == {2, 8, 9}
    assert d[2] == {'added': ['X'], 'removed': ['Z'], 'unchanged': ['Y']}
    assert d[8] == {'added': [], 'removed': ['Q'], 'unchanged': []}
    assert d[9] == {'added': ['P2e6'], 'removed': [], 'unchanged': []}

def test_diff_static_drops_self_consistent() -> None:
    sample = {2: ('LT2aA2', 'LT3a8'), 4: ('LT2aA11', 'LT2bHb12', 'LF3cIc10', 'LF3aOFa16'), 9: ('P2e6', 'P2e8')}
    d = guard1.diff_static_drops(sample, sample)
    assert set(d) == set(sample)
    for sid, rec in d.items():
        assert rec['added'] == [] and rec['removed'] == [], sid
        assert set(rec['unchanged']) == set(sample[sid])

def test_per_session_static_drops_no_aggregation() -> None:
    labels = ['a', 'b', 'c', 'd']
    rmad = [1.0, 1.0, 1.0, 1.0]
    sess = [_session(2, 0, labels, [0.02, 0.0, 0.0, 0.0], rmad), _session(2, 1, labels, [0.0, 0.02, 0.0, 0.0], rmad)]
    out = guard1.per_session_static_drops(sess)
    assert out == {(2, 0): ['a'], (2, 1): ['b']}

def test_per_session_static_drops_clean_session_is_empty() -> None:
    labels = ['a', 'b']
    out = guard1.per_session_static_drops([_session(4, 0, labels, [0.0, 0.0], [1.0, 1.0])])
    assert out == {(4, 0): []}

def test_per_session_static_drops_identity_guard_catches_drift() -> None:
    labels = ['a', 'b']
    rec = _session(9, 0, labels, [0.02, 0.0], [1.0, 1.0])
    rec['static'] = ['b']
    with pytest.raises(ValueError, match='threshold drift|threshold'):
        guard1.per_session_static_drops([rec])

def test_per_session_static_drops_identity_guard_passes_on_match() -> None:
    labels = ['a', 'b']
    rec = _session(9, 0, labels, [0.02, 0.0], [1.0, 1.0])
    rec['static'] = ['a']
    out = guard1.per_session_static_drops([rec])
    assert out == {(9, 0): ['a']}

def test_format_per_session_literal_roundtrips_via_literal_eval() -> None:
    import ast
    per_session = {(2, 0): ['RT1aIa7', 'RT1aIa8'], (2, 1): [], (9, 0): ['P2e6']}
    src = guard1.format_per_session_literal(per_session)
    body = src.split('=', 1)[1].strip()
    parsed = ast.literal_eval(body)
    assert parsed == {(2, 0): ('RT1aIa7', 'RT1aIa8'), (2, 1): (), (9, 0): ('P2e6',)}
    assert "(9, 0): ('P2e6',)," in src
    assert '(2, 1): (),' in src
    assert src.index('(2, 0)') < src.index('(2, 1)') < src.index('(9, 0)')
