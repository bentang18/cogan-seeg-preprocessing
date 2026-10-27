"""Tests for the localization sphere-sampler geometry (synthetic volume, no FS)."""
from __future__ import annotations
import csv
import os
import numpy as np
import pytest
import cogan_localize as cl
_D24_DIR = '/nonexistent/private-test-fixtures'

def test_sphere_offsets_counts_and_origin():
    o0 = cl.sphere_offsets(0)
    assert o0.tolist() == [[0, 0, 0]]
    o1 = cl.sphere_offsets(1)
    assert o1.shape == (7, 3)
    assert [0, 0, 0] in o1.tolist()
    assert [1, 1, 1] not in o1.tolist()

def test_sphere_offsets_radius_negative_raises():
    with pytest.raises(ValueError, match='radius_vox'):
        cl.sphere_offsets(-1)

def test_proportions_uniform_volume():
    vol = np.full((10, 10, 10), 5, dtype=np.int32)
    origin, props = cl.sphere_label_proportions(vol, (5, 5, 5), 2.0)
    assert origin == 5
    assert props == [(5, 1.0)]

def test_proportions_two_labels_split_by_plane():
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[:5] = 1
    vol[5:] = 2
    origin, props = cl.sphere_label_proportions(vol, (5, 5, 5), 1.0)
    assert origin == 2
    d = dict(props)
    assert set(d) == {1, 2}
    assert abs(sum(d.values()) - 1.0) < 1e-09
    assert abs(d[2] - 6 / 7) < 1e-09
    assert abs(d[1] - 1 / 7) < 1e-09

def test_proportions_sorted_desc():
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[:] = 3
    vol[5, 5, 5] = 9
    origin, props = cl.sphere_label_proportions(vol, (5, 5, 5), 2.0)
    assert origin == 9
    assert props[0][0] == 3
    assert props[0][1] > props[1][1]

def test_edge_electrode_skips_out_of_bounds():
    vol = np.full((10, 10, 10), 7, dtype=np.int32)
    origin, props = cl.sphere_label_proportions(vol, (0, 0, 0), 2.0)
    assert origin == 7
    assert props == [(7, 1.0)]

def test_center_out_of_bounds_origin_zero():
    vol = np.full((10, 10, 10), 7, dtype=np.int32)
    origin, props = cl.sphere_label_proportions(vol, (-5, 5, 5), 1.0)
    assert origin == 0
    assert all((lab in (0, 7) for lab, _ in props))

def test_rounding_of_fractional_center():
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[5, 5, 5] = 4
    origin, _ = cl.sphere_label_proportions(vol, (5.4, 4.6, 5.1), 0.0)
    assert origin == 4
_ELECNAMES = '12-Jun-2019 16:56:47\nName, Depth/Strip/Grid, Hem\nLTG8 G L\nROG13 D R\nL1IF10 D L\n'
_LEPTOVOX = '12-Jun-2019 16:56:47\tdykstra-\tfreesurfer-v6.0.0\nR A S\n191.974403 117.809466 146.057659\n100.5 110.25 120.0\n50.0 60.0 70.0\n'

def test_read_electrode_names_skips_headers_and_parses(tmp_path):
    p = tmp_path / 'D99.electrodeNames'
    p.write_text(_ELECNAMES)
    rows = cl.read_electrode_names(str(p))
    assert rows == [('LTG8', 'G', 'L'), ('ROG13', 'D', 'R'), ('L1IF10', 'D', 'L')]

def test_read_electrode_names_rejects_bad_row(tmp_path):
    p = tmp_path / 'bad.electrodeNames'
    p.write_text('hdr1\nhdr2\nLTG8 G L\nJUNKROW\n')
    with pytest.raises(ValueError, match='unparseable'):
        cl.read_electrode_names(str(p))

def test_read_leptovox_skips_headers_and_parses(tmp_path):
    p = tmp_path / 'D99.LEPTOVOX'
    p.write_text(_LEPTOVOX)
    coords = cl.read_leptovox(str(p))
    assert coords.shape == (3, 3)
    np.testing.assert_allclose(coords[0], [191.974403, 117.809466, 146.057659])
    np.testing.assert_allclose(coords[2], [50.0, 60.0, 70.0])

def test_read_leptovox_rejects_non_triple(tmp_path):
    p = tmp_path / 'bad.LEPTOVOX'
    p.write_text('h1\nh2\n1.0 2.0 3.0\n1.0 2.0\n')
    with pytest.raises(ValueError, match='not 3 floats'):
        cl.read_leptovox(str(p))

def test_read_recon_electrodes_aligned(tmp_path):
    en = tmp_path / 'D99.electrodeNames'
    lv = tmp_path / 'D99.LEPTOVOX'
    en.write_text(_ELECNAMES)
    lv.write_text(_LEPTOVOX)
    names, coords = cl.read_recon_electrodes(str(en), str(lv))
    assert [n[0] for n in names] == ['LTG8', 'ROG13', 'L1IF10']
    assert coords.shape == (3, 3)

def test_read_recon_electrodes_length_mismatch_raises(tmp_path):
    en = tmp_path / 'D99.electrodeNames'
    lv = tmp_path / 'D99.LEPTOVOX'
    en.write_text(_ELECNAMES)
    lv.write_text('h1\nh2\n1 2 3\n4 5 6\n')
    with pytest.raises(ValueError, match='recon mismatch'):
        cl.read_recon_electrodes(str(en), str(lv))
_POSTIMPLOC = 'LAT 14 182.000000 122.000000 112.000000 L D\nLAT 13 178.250000 122.000000 113.000000 L D\nLPT 16 50.000000 60.000000 70.000000 L D\nLPT 15 51.000000 61.000000 71.000000 L D\n'

def test_read_postimploc_names_concatenates_shaft_and_num(tmp_path):
    p = tmp_path / 'D59PostimpLoc.txt'
    p.write_text(_POSTIMPLOC)
    assert cl.read_postimploc_names(str(p)) == ['LAT14', 'LAT13', 'LPT16', 'LPT15']

def test_recon_postimploc_recovers_by_name_and_drops_extras(tmp_path):
    en = tmp_path / 'D59.electrodeNames'
    en.write_text('hdr1\nhdr2\nLAT14 D L\nLAT13 D L\n')
    lv = tmp_path / 'D59.LEPTOVOX'
    lv.write_text('hdr1\nhdr2\n182.0 122.0 112.0\n178.25 122.0 113.0\n50.0 60.0 70.0\n51.0 61.0 71.0\n')
    pl = tmp_path / 'D59PostimpLoc.txt'
    pl.write_text(_POSTIMPLOC)
    names, coords = cl.read_recon_electrodes_postimploc(str(en), str(lv), str(pl))
    assert [n[0] for n in names] == ['LAT14', 'LAT13']
    np.testing.assert_allclose(coords[0], [182.0, 122.0, 112.0])
    np.testing.assert_allclose(coords[1], [178.25, 122.0, 113.0])

def test_recon_postimploc_missing_electrode_raises(tmp_path):
    en = tmp_path / 'D59.electrodeNames'
    en.write_text('h1\nh2\nGHOST9 D L\n')
    lv = tmp_path / 'D59.LEPTOVOX'
    lv.write_text('h1\nh2\n1 2 3\n')
    pl = tmp_path / 'D59PostimpLoc.txt'
    pl.write_text('LAT 14 1 2 3 L D\n')
    with pytest.raises(ValueError, match='absent from PostimpLoc'):
        cl.read_recon_electrodes_postimploc(str(en), str(lv), str(pl))

def test_recon_postimploc_leptovox_length_mismatch_raises(tmp_path):
    en = tmp_path / 'D59.electrodeNames'
    en.write_text('h1\nh2\nLAT14 D L\n')
    lv = tmp_path / 'D59.LEPTOVOX'
    lv.write_text('h1\nh2\n1 2 3\n4 5 6\n')
    pl = tmp_path / 'D59PostimpLoc.txt'
    pl.write_text('LAT 14 1 2 3 L D\n')
    with pytest.raises(ValueError, match='not row-aligned'):
        cl.read_recon_electrodes_postimploc(str(en), str(lv), str(pl))
_LUT = '#No. Label Name:  R G B A\n\n0   Unknown                       0 0 0 0\n2   Left-Cerebral-White-Matter    245 245 245 0\n17  Left-Hippocampus              220 216 20 0\n1024    ctx-lh-precentral         60 20 220 0\n1030    ctx-lh-superiortemporal   140 220 220 0\n'

def test_read_fs_color_lut(tmp_path):
    p = tmp_path / 'lut.txt'
    p.write_text(_LUT)
    lut = cl.read_fs_color_lut(str(p))
    assert lut[0] == 'Unknown'
    assert lut[2] == 'Left-Cerebral-White-Matter'
    assert lut[17] == 'Left-Hippocampus'
    assert lut[1024] == 'ctx-lh-precentral'
    assert lut[1030] == 'ctx-lh-superiortemporal'

def test_label_ids_to_names_resolves_and_preserves_order():
    lut = {1024: 'ctx-lh-precentral', 2: 'Left-Cerebral-White-Matter'}
    origin_name, props = cl.label_ids_to_names(1024, [(1024, 0.7), (2, 0.2), (9999, 0.1)], lut)
    assert origin_name == 'ctx-lh-precentral'
    assert props == [('ctx-lh-precentral', 0.7), ('Left-Cerebral-White-Matter', 0.2), ('label-9999', 0.1)]

def test_write_depth_wm_csv_origin_choice(tmp_path):
    out = tmp_path / 'depth-wm.csv'
    rows = [('LTG8', 'G', 'L'), ('ROG13', 'D', 'R')]
    coords = np.array([[191.9, 117.8, 146.0], [100.5, 110.25, 120.0]])
    cl.write_depth_wm_csv(str(out), rows, coords, origin_names=['ctx-lh-supramarginal', 'Left-Cerebral-White-Matter'], majority_names=['ctx-lh-supramarginal', 'ctx-lh-superiortemporal'], majority_props=[0.6992, 0.5691], label_choice='origin', radius_mm=3.0)
    with open(out) as fh:
        got = list(csv.DictReader(fh))
    assert [r['Electrode'] for r in got] == ['LTG8', 'ROG13']
    assert got[0]['DKT'] == 'ctx-lh-supramarginal'
    assert got[1]['DKT'] == 'Left-Cerebral-White-Matter'
    assert got[1]['DKT_origin'] == 'Left-Cerebral-White-Matter'
    assert got[1]['DKT_majority'] == 'ctx-lh-superiortemporal'
    assert got[0]['radius_mm'] == '3.000'

def test_write_depth_wm_csv_majority_choice(tmp_path):
    out = tmp_path / 'depth-wm.csv'
    rows = [('LTG16', 'G', 'L')]
    coords = np.array([[1.0, 2.0, 3.0]])
    cl.write_depth_wm_csv(str(out), rows, coords, origin_names=['Left-Cerebral-White-Matter'], majority_names=['ctx-lh-superiortemporal'], majority_props=[0.51], label_choice='majority', radius_mm=3.0)
    with open(out) as fh:
        got = list(csv.DictReader(fh))
    assert got[0]['DKT'] == 'ctx-lh-superiortemporal'

def test_write_depth_wm_csv_bad_choice_raises(tmp_path):
    with pytest.raises(ValueError, match='label_choice'):
        cl.write_depth_wm_csv(str(tmp_path / 'x.csv'), [('A', 'D', 'L')], np.zeros((1, 3)), origin_names=['x'], majority_names=['y'], majority_props=[1.0], label_choice='center', radius_mm=3.0)

def test_write_depth_wm_csv_ragged_raises(tmp_path):
    with pytest.raises(ValueError, match='ragged'):
        cl.write_depth_wm_csv(str(tmp_path / 'x.csv'), [('A', 'D', 'L')], np.zeros((2, 3)), origin_names=['x'], majority_names=['y'], majority_props=[1.0], label_choice='origin', radius_mm=3.0)

def test_apply_axis_convention_identity_and_flip():
    coords = np.array([[1.0, 2.0, 3.0]])
    shape = (10, 10, 10)
    ident = cl.apply_axis_convention(coords, (0, 1, 2), (False, False, False), shape)
    np.testing.assert_allclose(ident, [[1, 2, 3]])
    perm = cl.apply_axis_convention(coords, (2, 0, 1), (True, False, False), shape)
    np.testing.assert_allclose(perm, [[6, 1, 2]])

def test_find_axis_convention_recovers_planted():
    rng = np.arange(8).reshape(2, 2, 2)
    vol = np.kron(rng, np.ones((5, 5, 5), dtype=np.int64)).astype(np.int64) + 1000
    lut = {1000 + v: f'lab-{v}' for v in range(8)}
    true_perm, true_flips = ((2, 0, 1), (True, False, True))
    ijk = np.array([[2, 2, 2], [7, 2, 7], [2, 7, 3], [7, 7, 7], [3, 3, 8]], float)
    shape = vol.shape
    recon = np.zeros_like(ijk)
    inv = np.argsort(true_perm)
    tmp = ijk.copy()
    for a in range(3):
        if true_flips[a]:
            tmp[:, a] = shape[a] - 1 - tmp[:, a]
    recon = tmp[:, inv]
    expected = [lut[int(vol[int(p[0]), int(p[1]), int(p[2])])] for p in ijk]
    perm, flips, frac = cl.find_axis_convention(vol, recon, expected, lut, 0.0)
    assert frac == 1.0
    assert perm == true_perm and flips == true_flips

def test_find_axis_convention_length_mismatch_raises():
    vol = np.zeros((4, 4, 4), dtype=np.int64)
    with pytest.raises(ValueError, match='length mismatch'):
        cl.find_axis_convention(vol, np.zeros((3, 3)), ['a', 'b'], {}, 0.0)

def test_gm_ids_from_lut_picks_in_vocab_only():
    lut = {0: 'Unknown', 2: 'Left-Cerebral-White-Matter', 17: 'Left-Hippocampus', 1024: 'ctx-lh-precentral', 1030: 'ctx-lh-superiortemporal', 1004: 'ctx-lh-corpuscallosum'}
    ids = cl.gm_ids_from_lut(lut, vocab=('ctx-lh-precentral', 'Left-Hippocampus'))
    assert ids == {1024, 17}
    d = cl.gm_ids_from_lut(lut)
    assert 1030 in d and 1024 in d and (17 in d)
    assert 0 not in d and 2 not in d and (1004 not in d)

def test_nearest_gm_center_in_gm_reach_zero():
    vol = np.full((12, 12, 12), 2, dtype=np.int64)
    vol[6, 6, 6] = 1024
    gm_id, reach, nbase = cl.nearest_gm_label(vol, (6, 6, 6), {1024}, 3.0, 10.0)
    assert gm_id == 1024
    assert reach == 0.0
    assert nbase == 1

def test_nearest_gm_plurality_ignores_white_matter():
    vol = np.full((14, 14, 14), 2, dtype=np.int64)
    vol[6:9, 6:9, 6:9] = 1030
    gm_id, reach, nbase = cl.nearest_gm_label(vol, (7, 7, 7), {1024, 1030}, 3.0, 10.0)
    assert gm_id == 1030
    assert reach == 0.0
    assert nbase == 27

def test_nearest_gm_grows_past_base_to_reach_gm():
    vol = np.full((14, 14, 14), 2, dtype=np.int64)
    vol[6, 6, 11] = 1024
    gm_id, reach, nbase = cl.nearest_gm_label(vol, (6, 6, 6), {1024}, 3.0, 10.0)
    assert gm_id == 1024
    assert abs(reach - 5.0) < 1e-09
    assert nbase == 0

def test_nearest_gm_deep_wm_beyond_cap_returns_sentinel():
    vol = np.full((26, 26, 26), 2, dtype=np.int64)
    vol[6, 6, 17] = 1024
    gm_id, reach, nbase = cl.nearest_gm_label(vol, (6, 6, 6), {1024}, 3.0, 10.0)
    assert gm_id == 0
    assert reach == float('inf')
    assert nbase == 0

def test_nearest_gm_tie_breaks_to_nearer_parcel():
    vol = np.full((14, 14, 14), 2, dtype=np.int64)
    vol[6, 6, 7] = 1024
    vol[6, 6, 8] = 1024
    vol[6, 4, 6] = 1030
    vol[6, 3, 6] = 1030
    gm_id, _, _ = cl.nearest_gm_label(vol, (6, 6, 6), {1024, 1030}, 3.0, 10.0)
    assert gm_id == 1024

def test_nearest_gm_invalid_radii_raise():
    vol = np.zeros((6, 6, 6), dtype=np.int64)
    with pytest.raises(ValueError, match='r_base_vox'):
        cl.nearest_gm_label(vol, (3, 3, 3), {1}, -1.0, 10.0)
    with pytest.raises(ValueError, match='r_base_vox'):
        cl.nearest_gm_label(vol, (3, 3, 3), {1}, 5.0, 3.0)

def test_write_depth_wm_csv_nearest_gm_choice_and_counterfactual_columns(tmp_path):
    out = tmp_path / 'depth-wm.csv'
    rows = [('A1', 'D', 'L'), ('A2', 'D', 'L')]
    coords = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    cl.write_depth_wm_csv(str(out), rows, coords, origin_names=['Left-Cerebral-White-Matter', 'Left-Cerebral-White-Matter'], majority_names=['Left-Cerebral-White-Matter', 'ctx-lh-superiortemporal'], majority_props=[0.7, 0.55], label_choice='nearest_gm', radius_mm=3.0, nearest_gm_names=['ctx-lh-precentral', 'unknown'], gm_reach_mm=[1.4142, float('inf')], n_gm_base=[8, 0], cap_mm=10.0)
    with open(out) as fh:
        got = list(csv.DictReader(fh))
    assert got[0]['DKT'] == 'ctx-lh-precentral'
    assert got[1]['DKT'] == 'unknown'
    assert got[0]['DKT_origin'] == 'Left-Cerebral-White-Matter'
    assert got[0]['DKT_majority'] == 'Left-Cerebral-White-Matter'
    assert got[0]['DKT_nearest_gm'] == 'ctx-lh-precentral'
    assert got[0]['gm_reach_mm'] == '1.4142'
    assert got[1]['gm_reach_mm'] == 'inf'
    assert got[0]['n_gm_base'] == '8'
    assert got[0]['cap_mm'] == '10.000'

def test_write_depth_wm_csv_nearest_requires_columns(tmp_path):
    with pytest.raises(ValueError, match='requires nearest_gm_names'):
        cl.write_depth_wm_csv(str(tmp_path / 'x.csv'), [('A', 'D', 'L')], np.zeros((1, 3)), origin_names=['x'], majority_names=['y'], majority_props=[1.0], label_choice='nearest_gm', radius_mm=3.0)
