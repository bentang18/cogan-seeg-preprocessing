"""Apply the existing native-volume localization helper. Historical axis calibration uses its reference export; verify compatibility before applying to new reconstructions."""
from __future__ import annotations
import argparse
import os
import re
import numpy as np
import cogan_localize as cl
PIN_SUBJECTS = ('D23',)
_SHAFT_RE = re.compile('^(?P<shaft>.*?)(?P<depth>\\d+)$')

def _shaft_prefix(name: str) -> str:
    m = _SHAFT_RE.match(name)
    return m.group('shaft') if m else name

def _recon_dir(recon_root: str, subj: str) -> str:
    return os.path.join(recon_root, subj, 'elec_recon')

def _mri_dir(recon_root: str, subj: str) -> str:
    return os.path.join(recon_root, subj, 'mri')

def _load_mgz(path: str) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(path).get_fdata()).astype(np.int64)

def _dk_csv_path(recon_root: str, subj: str, radius_mm: float) -> str:
    r = int(round(radius_mm))
    return os.path.join(_recon_dir(recon_root, subj), f'{subj}_elec_location_radius_{r}mm_aparc+aseg.mgz.csv')

def pin_convention(recon_root: str, lut: dict[int, str], radius_mm: float, threshold: float) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Reproduce the depth-reference DK CSV to lock the LEPTOVOX→array convention.

    The winning ``(perm, flips)`` must clear ``threshold`` on every pin subject
    (depth-only; see ``PIN_SUBJECTS``), else we abort — a match below threshold
    means the geometry is unresolved. Multiple pin subjects must also agree.
    """
    winner: tuple[tuple[int, ...], tuple[bool, ...]] | None = None
    for subj in PIN_SUBJECTS:
        rd = _recon_dir(recon_root, subj)
        names, coords = cl.read_recon_electrodes(os.path.join(rd, f'{subj}.electrodeNames'), os.path.join(rd, f'{subj}.LEPTOVOX'))
        gt = cl.read_dk_csv_origins(_dk_csv_path(recon_root, subj, radius_mm))
        if len(gt) != len(names):
            raise SystemExit(f'{subj}: DK CSV rows ({len(gt)}) != recon electrodes ({len(names)}) — cannot row-align the pin ground truth.')
        mismatched = [(i, n[0], gt[i][0]) for i, n in enumerate(names) if _shaft_prefix(n[0]) != _shaft_prefix(gt[i][0])]
        if mismatched:
            raise SystemExit(f'{subj}: DK CSV is NOT row-aligned to the recon — shaft prefix differs at rows {mismatched[:5]}; row-index pin is unsafe.')
        expected = [origin for _, origin in gt]
        vol = _load_mgz(os.path.join(_mri_dir(recon_root, subj), 'aparc+aseg.mgz'))
        perm, flips, frac = cl.find_axis_convention(vol, coords, expected, lut, radius_mm)
        print(f'[pin] {subj}: convention perm={perm} flips={flips} match={frac:.4f}')
        if frac < threshold:
            raise SystemExit(f'axis-convention pin FAILED on {subj}: best match {frac:.4f} < threshold {threshold} — geometry unresolved, aborting.')
        if winner is None:
            winner = (perm, flips)
        elif (perm, flips) != winner:
            raise SystemExit(f'pin subjects disagree on convention: {winner} vs {(perm, flips)}')
    assert winner is not None
    return winner

def localize_subject(recon_root: str, subj: str, lut: dict[int, str], perm: tuple[int, ...], flips: tuple[bool, ...], radius_mm: float, cap_mm: float, label_choice: str, out_path: str, depth_only: bool=True) -> int:
    """Sample the DKT atlas for one subject → write ``depth-wm.csv``. Returns N.

    ``depth_only`` (default True) keeps only kind ``D`` (sEEG depth) contacts and
    drops grid/strip (``G``/``S``) surface contacts: the Cogan v3 feed is
    depth-only, center-voxel volume labeling is unreliable for surface contacts
    (~0.44 reproduction), and 2D grids have no linear shaft for the depth-RoPE.
    A pure-surface subject then yields an empty ``depth-wm.csv`` (N=0) and drops
    out of the feed naturally; a mixed subject (e.g. D67) keeps only its depths.
    """
    rd = _recon_dir(recon_root, subj)
    en_path = os.path.join(rd, f'{subj}.electrodeNames')
    lv_path = os.path.join(rd, f'{subj}.LEPTOVOX')
    try:
        names, coords = cl.read_recon_electrodes(en_path, lv_path)
    except ValueError:
        pl_path = os.path.join(rd, f'{subj}PostimpLoc.txt')
        if not os.path.isfile(pl_path):
            raise
        names, coords = cl.read_recon_electrodes_postimploc(en_path, lv_path, pl_path)
        print(f'[recover] {subj}: name-aligned {len(names)} contacts via PostimpLoc')
    if depth_only:
        keep = [k for k, n in enumerate(names) if n[1] == 'D']
        names = [names[k] for k in keep]
        coords = coords[keep]
    vol = _load_mgz(os.path.join(_mri_dir(recon_root, subj), 'aparc.DKTatlas+aseg.mgz'))
    ijk = cl.apply_axis_convention(coords, perm, flips, vol.shape)
    gm_ids = cl.gm_ids_from_lut(lut)
    origin_names: list[str] = []
    majority_names: list[str] = []
    majority_props: list[float] = []
    nearest_gm_names: list[str] = []
    gm_reach_mm: list[float] = []
    n_gm_base: list[int] = []
    for k in range(ijk.shape[0]):
        origin_id, props = cl.sphere_label_proportions(vol, ijk[k], radius_mm)
        origin_name, named = cl.label_ids_to_names(origin_id, props, lut)
        origin_names.append(origin_name)
        if named:
            majority_names.append(named[0][0])
            majority_props.append(named[0][1])
        else:
            majority_names.append('unknown')
            majority_props.append(0.0)
        gm_id, reach, nbase = cl.nearest_gm_label(vol, ijk[k], gm_ids, radius_mm, cap_mm)
        nearest_gm_names.append(lut.get(gm_id, 'unknown') if gm_id else 'unknown')
        gm_reach_mm.append(reach)
        n_gm_base.append(nbase)
    cl.write_depth_wm_csv(out_path, names, coords, origin_names, majority_names, majority_props, label_choice=label_choice, radius_mm=radius_mm, nearest_gm_names=nearest_gm_names, gm_reach_mm=gm_reach_mm, n_gm_base=n_gm_base, cap_mm=cap_mm)
    return len(names)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recon-root', required=True)
    ap.add_argument('--lut', required=True, help='FreeSurferColorLUT.txt')
    ap.add_argument('--subjects', nargs='+', required=True, help='e.g. D23 D24 D19 …')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--radius-mm', type=float, default=3.0, help="recording-sphere radius (mm=vox on the 1mm conformed volume). 3.0 = the lab's own ECoG_Recon neighborhood (shipped DK CSV is radius_3mm), so the nearest-GM vote runs over the same sphere the lab sampled.")
    ap.add_argument('--cap-mm', type=float, default=10.0, help="Maximum reach for the inherited nearest-GM search (mm). No eligible gray voxel within this cap returns unknown; this does not identify why the search failed or certify localization accuracy.")
    ap.add_argument('--pin-threshold', type=float, default=0.9, help='Minimum reference-export origin-label agreement for the historical axis calibration. Inspect disagreements; passing is not proof of anatomical accuracy or correct acquisition-channel identity.')
    ap.add_argument('--keep-surface', action='store_true', help='also localize grid/strip (G/S) contacts (default: depth-only). For auditing surface reproduction only — the v3 feed drops surface contacts.')
    ap.add_argument('--label-choice', choices=('nearest_gm', 'majority', 'origin'), default='nearest_gm', help="Which candidate fills the DKT column. The inherited nearest_gm rule votes among in-vocabulary gray voxels, growing the base sphere if needed up to --cap-mm. Origin and majority candidates are retained for audit. Choose and verify the anatomical policy explicitly.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lut = cl.read_fs_color_lut(args.lut)
    perm, flips = pin_convention(args.recon_root, lut, args.radius_mm, args.pin_threshold)
    print(f'[pin] LOCKED convention perm={perm} flips={flips}')
    for subj in args.subjects:
        out = os.path.join(args.out_dir, f'{subj}_depth-wm.csv')
        try:
            n = localize_subject(args.recon_root, subj, lut, perm, flips, args.radius_mm, args.cap_mm, args.label_choice, out, depth_only=not args.keep_surface)
            tag = 'ok' if n else 'EMPTY(surface-only)'
            print(f'[{tag}] {subj}: {n} depth electrodes → {out}')
        except (FileNotFoundError, ValueError) as e:
            print(f'[SKIP] {subj}: {e}')
if __name__ == '__main__':
    main()
