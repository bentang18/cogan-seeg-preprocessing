# Cogan sEEG preprocessing

Small research tools for preparing Cogan Lab's D-cohort sEEG recordings. Start
here without access to the private model-training repository.

This is a **starter toolkit, not a fully validated all-task pipeline**. It reuses
the existing inventory, voltage loader, bad-contact detector, native-volume
localization helpers and cached bad-span detector. Detector thresholds and atlas
assignment rules are inherited research choices, not guarantees that a recording
is clean or a contact is correctly localized.

## Start here

```bash
uv sync --extra dev
uv run pytest -q
```

1. Compare the available Box task folders with DCC. **Box has the newest lab
   data; the DCC copy may lag.** Keep the source files unchanged. Put derived
   reports and working copies in your own approved workspace.
2. Pick one subject/task. Check EDF headers, channel names/types, reference
   channels, electrode coordinates and recording clocks before processing.
3. Run the bad-contact scan, inspect raw traces and its proposed exclusions, then
   apply approved exclusions **before** referencing or computing frontend caches.
4. Match neural channels to localized contacts by verified identities. Plot the
   locations against the subject's native MRI/surfaces and inspect suspicious joins.
5. Once compatible frontend caches exist, inspect proposed bad spans alongside
   retained intervals. Keep a short list of missing files, ambiguities and changes.

Run signal processing in a compute allocation, not on the login node. Lab source
storage may be login-node-only: stage read-only copies into compute-visible
personal storage and update the manifest paths before launching jobs.

## Tools

| File | Purpose |
| --- | --- |
| `scripts/build_cogan_manifest.py` | Historical seven-task BIDS run inventory, including source paths and rates |
| `src/cogan_preprocessing/loader.py` | EDF read, neural-channel selection, identity-preserving row order and resampling |
| `scripts/precompute_guard1_static_cogan.py` | Pre-reference bad-contact scan, one manifest run at a time |
| `scripts/collect_guard1_static_cogan.py` | Collect decisions with optional full-manifest coverage enforcement |
| `scripts/cogan_localize.py` | Native-coordinate parsers and volumetric atlas-sampling primitives |
| `scripts/localize_cogan_batch.py` | Existing localization driver with historical reference-based axis calibration |
| `scripts/precompute_bad_windows_cogan.py` | Bad-span scan on existing compatible three-band caches |

### Inventory and one-run contact review

```bash
uv run python scripts/build_cogan_manifest.py \
  --data-root /path/to/staged-task-folders \
  --out /path/to/personal-outputs/manifest.csv

uv run python scripts/precompute_guard1_static_cogan.py \
  --manifest /path/to/personal-outputs/manifest.csv \
  --row 0 --out-dir /path/to/personal-outputs/contact-detections
```

The manifest builder's `PLANNED_TASKS` is a fixed historical list of seven tasks.
It is **not** a full inventory of today's Box collection. Review additional tasks,
duplicate exports and versioned folders explicitly. Its current discovery picks
the first sorted matching version, so do not treat that as the latest or canonical
version. Missing sidecars, unsupported implant suffixes and skipped files need
review rather than silent exclusion.

After scanning every intended manifest row:

```bash
uv run python scripts/collect_guard1_static_cogan.py \
  --detections-dir /path/to/personal-outputs/contact-detections \
  --manifest /path/to/personal-outputs/manifest.csv \
  --out /path/to/personal-outputs/contact-exclusions.json
```

The collector checks completeness. The exclusion JSON is a derived artifact;
it does not alter an EDF or automatically approve all proposed channel drops.

### Anatomy

```bash
uv run python scripts/localize_cogan_batch.py --help
```

The driver consumes FreeSurfer volumes and the existing `electrodeNames`,
`LEPTOVOX` and optional `PostimpLoc` recon files. Its historical axis calibration
uses a reference DK export, with `PIN_SUBJECTS` declared in the script. This is
**not** a universal transform for arbitrary new BrainTRACE/BIDS exports. Verify
the native coordinate frame and channel/contact correspondence first. It
defaults to depth contacts, with other contact kinds handled separately.

Outputs retain origin/sphere candidates and nearest-gray-matter reach information.
The inherited nearest-GM rule defaults to a 3 mm base sphere and 10 mm reach cap.
These are explicit assumptions for review, not localization accuracy bounds.
Changing the atlas, snapping rule or thresholds changes the preprocessing contract.

### Frontend caches and bad spans

The private training dispatcher and its cache-build submitter are intentionally
**not included**. They require the larger experiment stack. This repository does
not offer a replacement cache builder or claim end-to-end cache parity.

The released [MAPA frontend](https://github.com/bentang18/MAPA/tree/main/mapa/preprocessing)
is available separately; integrating its output requires a checked format/recipe
adapter. Do not point the scanner at an arbitrary output directory and assume
its metadata or normalization conventions match.

For an already compatible legacy cache:

```bash
uv run python scripts/precompute_bad_windows_cogan.py \
  --spec-cache-dir /path/to/compatible-cache \
  --manifest /path/to/staged-manifest.csv \
  --out-dir /path/to/personal-outputs/bad-spans
```

Expected bands are `band_v3slow`, `band_v3mid`, `band_hga`, all on the same
32 Hz frame grid. Each cache has raw magnitude `(contacts, bins, frames)` `.npy`,
matching channel-order/rate `.json`, and median/sigma `.stats.npz` files. This
scanner is not for the newer mixed-rate uECoG caches. It reads the already rebuilt
post-contact-exclusion domain and detects before read-time clipping.

## What has been checked

Tests use generated arrays, synthetic cache sidecars and temporary recon-format
files. They cover channel selection/order, resampling, static-contact decisions,
collection coverage, atlas arithmetic and injected cache artifacts. A synthetic
EDF round trip checks the real MNE reader/scan boundary. These checks do not
certify any new participant data, clinical localization or artifact threshold.

`SOURCE_PROVENANCE.json` records where each copied implementation came from.
Changes are packaging/import boundaries and explicit source-root arguments.
The numerical detector and atlas primitives are retained from the existing code.

No EDFs, MRI scans, recordings, subject manifests, per-subject coordinates,
contact-exclusion maps, credentials or private repository history are included.
