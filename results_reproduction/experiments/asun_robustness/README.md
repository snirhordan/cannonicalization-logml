# ASUN robustness harness

Adversarial test of whether ASUN (Baker et al. 2024, ICML, "An Explicit
Frame Construction for Normalizing 3D Point Clouds") produces unstable
canonical frames on near-symmetric molecular point clouds, versus our
finite-group canonicalization (`canonicalize_3d` / `canonicalize_3d_wwv`).

## How to run

```bash
# one-time: pymatgen into gnnplus (already done for this session)
conda run -n gnnplus pip install pymatgen

# metrics self-test (sigma=0 consistency check)
conda run -n gnnplus python3 metrics.py

# methods smoke test
conda run -n gnnplus python3 methods.py

# pilot (local validation)
conda run -n gnnplus python3 run_sweep.py --mode pilot --cores 2 \
    --out results/pilot.csv --qm9-pilot-scan 3000 --nperturb 15 --seed 0

# full sweep (orchestrator submits, NOT run here)
sbatch submit_full.sbatch
# equivalent direct command the sbatch script runs:
conda run -n gnnplus python3 run_sweep.py --mode full --cores 64 \
    --out results/full.csv --sigmas 0,1e-4,1e-3,3e-3,1e-2,3e-2,1e-1 \
    --nperturb 100 --seed 0
```

## Files

- `methods.py` -- uniform wrappers: `asun` (CatFrame, species, tol=1e-2
  fixed), `pca` (sign-fixed PCA, species-blind), `ours`
  (`canonicalize_3d`, species-blind), `ours_wwv` (`canonicalize_3d_wwv`,
  species-blind). Every wrapper returns `MethodResult(pose, species_out,
  crashed, err_type, regime, n_candidates)`, never raises.
- `metrics.py` -- species-constrained EMD (Hungarian assignment) +
  frame-residual Procrustes angle theta + flip-rate (theta > 5 deg).
  `__main__` self-test.
- `data_prep.py` -- QM9 loading + pymatgen point-group labeling (tol=0.1,
  cached to `results/qm9_point_group_labels.csv`) + the synthetic
  symmetric battery + QM-sym opportunistic download/parsing.
- `run_sweep.py` -- CLI sweep driver, multiprocessing.Pool over
  (molecule, perturbation) tasks. Writes `<out>.csv` (tidy aggregate) and
  `<out>_raw_per_molecule.csv` (per-molecule aggregate).
- `submit_full.sbatch` -- SLURM script for the full run (public,
  `--account=cs`, `-c 64`, CPU-only). **Not submitted by this harness.**
- `vendor/pyorbit/` -- a vendored copy of ASUN
  (github.com/Utah-Math-Data-Science/alignment, commit `17d2c7f`,
  2024-07-25), so the harness does not depend on the orchestrator's
  ephemeral job tmp directory (`/home/.../jobs/<id>/tmp/asun_peek`), which
  may not exist by the time the full SLURM job runs.

## Fairness notes

- Identical `Q_p` (Haar-random rotation) and `Z_p` (unit Gaussian noise,
  scaled by sigma) for a given `(molecule, p)` across all 4 methods: one
  `np.random.RandomState(seed)` per `(molecule.name, p, --seed)` (a
  deterministic CRC32-derived seed, stable across processes), consumed in
  the fixed order `Rotation.random(...)` then `standard_normal(...)`.
  Noise is added to the perturbed copy only (`baseline = X`, `perturbed =
  X @ Q_p.T + sigma * Z_p`); sigma=0 exactly zeroes the noise term rather
  than being a separate code path.
- ASUN always receives species (its strongest, most symmetry-reducing
  form); ours never does (`canonicalize_3d`/`_wwv` are geometry-only by
  design). We did not add species to ours, and did not tune ours's
  `decimals`/`gap_rtol`/`axial_rtol` away from the module defaults.
  ASUN's `tol=1e-2` is its own shipped default and is never tuned either.
- Crashes are counted as crashes (never as EMD=0 or a dropped row): a
  `crashed` boolean row is emitted, EMD/theta are NaN for it, and
  crash_rate is aggregated separately from the EMD/theta means (which are
  computed over non-crashed rows only). Baseline canonicalization is
  computed once per (molecule, method) since all four methods are
  deterministic pure functions -- exactly equivalent to recomputing it
  per-perturbation, at a fraction of the cost -- and if the CLEAN baseline
  itself crashes, every perturbation for that (molecule, method) is
  correctly recorded as crashed too (not silently skipped).
- Per-stratum molecule counts (`n_molecules`) are always in the tidy CSV
  and printed by `run_sweep.py`, so thin strata are visible rather than
  hidden inside an average.

## Synthetic symmetric battery

Regular polygons n=3..8, bipyramids n=3..8, prisms n=3..8 (edge=1.4 A),
Platonic solids (tetrahedron Td, cube Oh, octahedron Oh, icosahedron Ih),
and idealized real molecules (benzene D6h, 1,3,5-triazine D3h, methane Td,
allene D2d, CO2 D*h, acetylene D*h, trans-1,2-dichloroethylene C2h). Every
geometry's point group is **verified**, not assumed: each one is run
through the identical `pymatgen.symmetry.analyzer.PointGroupAnalyzer(tol=
0.1)` call used for QM9, and every one of the 29 entries matched its
intended design label exactly (details in `data_prep.py`'s
`build_synthetic_battery`).

## QM-sym: opportunistic download SUCCEEDED (better than expected)

The task brief said to try QM-sym for <=2 minutes and fall back to the
synthetic battery if it failed. It did not fail: `try_qm_sym_download`
retrieved the figshare "download all files" bundle
(`ndownloader.figshare.com/articles/9638093/versions/1`, ~374 MB) inside
the budget. We wrote a parser for QM-sym's bespoke `.xyz` format (line 1 =
atom count, line 2 = pipe-delimited properties whose first field is the
**declared symmetry group**, then `Symbol x y z mulliken_charge` per atom)
and reservoir-sample a configurable number of molecules per
`QM_sym_xyz_N.tar` shard (`build_qm_sym_sample`). We cross-validated a
random sample of the extracted molecules against our own pymatgen(tol=0.1)
labeling: 7/8 matched exactly; the one mismatch (file declared C2h,
pymatgen detected D2h) is consistent with the dataset's own documented
caveat that DFT-relaxed QM-sym geometries only *approximately* satisfy
their declared symmetry. We use the file's OWN declared label (ground
truth per QM-sym's symmetry-constrained construction), not a re-derived
one, as `point_group` for QM-sym records. This gives real, DFT-quality
C2h/C3h/C4h molecules for the pilot and full run, in addition to (not
instead of) the synthetic battery.

## Validation results (pilot: 3000 QM9 molecules scanned, ~8/stratum
sampled + whole synthetic battery + 3-500/tar QM-sym, nperturb=15)

### Gate (sigma=0 rotation-only consistency)

On **generic / non-symmetric** data (QM9 C1 molecules, and ours/PCA on
every stratum including the most symmetric synthetic shapes), all methods
are consistent at sigma=0 to floating-point precision: EMD ~1e-15, theta
~1e-7 deg. This confirms the harness itself (Q_p/Z_p pairing, species
alignment, Procrustes formula, aggregation) has no bug -- if it did, ours
and PCA would not be this exact.

**ASUN, however, already fails its own rotation-invariance gate at
sigma=0 on near-symmetric strata** -- this is not a harness bug (ours/PCA
on the SAME molecules, SAME Q_p, are exact), it is the first, earliest
sign of the phenomenon under test:

| method | dataset | point_group | n_mol | sigma=0 crash_rate | mean EMD | mean theta (deg) |
|---|---|---|---|---|---|---|
| asun | qm9 | C1  | 8 | 0.0 | 1.1e-15 | 5.5e-7 |
| asun | qm9 | C2v | 8 | 0.0 | 0.169   | 7.51   |
| ours | qm9 | C1  | 8 | 0.0 | 1.5e-15 | 3.9e-7 |
| ours | qm9 | C2v | 8 | 0.0 | 3.2e-15 | 3.7e-7 |
| pca  | qm9 | C2v | 8 | 0.0 | 3.2e-15 | 4.8e-7 |

### Effect (crash rate + theta grow with sigma, on symmetric strata)

**Crash resilience -- the cleanest, metric-artifact-free result.** ASUN
crashes (AssertionError) on ALL bipyramids (n=3..8) and the octahedron,
**at every sigma we tested, including sigma=0 (pure rotation, zero
noise)**: crash_rate = 1.0 unconditionally. We traced the exact assertion:
it is `assert v2 is not None, 'v2 is None'` in `CatFrame.v2_subroutine`
(not the convex-hull assertion) -- ASUN's anchor-frame construction
requires 3 pairwise NON-orthogonal reference directions
(`|dot(s_i,s_j)| > tol`), which provably cannot be found when a shape's
principal directions are mutually orthogonal (cubic/octahedral symmetry),
so the crash is structural, not merely a numerical tie. `ours`/`ours_wwv`
never crash on any input in the pilot (crash_rate = 0.0 everywhere).

**Theta/flip-rate on the majority "sign-ambiguity-only" regime (QM9
C1/Cs/C2/C2v/C2h and QM-sym's dominant C2h stratum, 60.9% of QM-sym) --
where `ours` gracefully degrades and ASUN does not:**

| method | dataset | point_group | sigma | crash_rate | mean EMD | mean theta (deg) | flip_rate |
|---|---|---|---|---|---|---|---|
| asun | qm9 | C2v | 0.000 | 0.0 | 0.169 | 7.51 | 0.083 |
| asun | qm9 | C2v | 0.001 | 0.0 | 0.218 | 10.15 | 0.117 |
| asun | qm9 | C2v | 0.010 | 0.0 | 0.342 | 16.12 | 0.250 |
| asun | qm9 | C2v | 0.100 | 0.0 | 1.011 | 36.77 | 0.975 |
| ours | qm9 | C2v | 0.000 | 0.0 | 3.2e-15 | 3.7e-7 | 0.000 |
| ours | qm9 | C2v | 0.001 | 0.0 | 5.3e-3 | 0.179 | 0.000 |
| ours | qm9 | C2v | 0.010 | 0.0 | 0.093 | 1.947 | 0.100 |
| ours | qm9 | C2v | 0.100 | 0.0 | 0.472 | 9.376 | 0.525 |
| asun | qm-sym | C2h | 0.000 | 0.0 | 1.288 | 22.23 | 0.317 |
| asun | qm-sym | C2h | 0.010 | 0.0 | 3.767 | 55.05 | 0.842 |
| ours | qm-sym | C2h | 0.000 | 0.0 | 3.6e-15 | 4.9e-7 | 0.000 |
| ours | qm-sym | C2h | 0.010 | 0.0 | 0.016 | 0.147 | 0.000 |

**Crash rate on the octahedron stratum (unconditional, all sigma):**

| method | dataset | point_group | sigma | crash_rate |
|---|---|---|---|---|
| asun | synthetic | Oh (mean of cube+octahedron) | any | 0.5 (octahedron alone: 1.0) |
| ours | synthetic | Oh | any | 0.0 |
| ours_wwv | synthetic | Oh | any | 0.0 |
| pca | synthetic | Oh | any | 0.0 |

### An honest complication: true spectral-degeneracy strata (C3h, C4h,
cubic/icosahedral Platonic solids, exactly-linear D*h/C*v)

This is the one place we did NOT observe a clean "ours wins" effect, and
we report it exactly as found rather than hiding it:

1. **Exactly-symmetric idealized molecules (our own synthetic benzene,
   methane, etc.) immediately destabilize `ours`/`ours_wwv` too**, at the
   smallest sigma tested (1e-4), reaching ~100% flip-rate. E.g. synthetic
   methane (Td): `ours` theta jumps from 0 (sigma=0) to 53.3 deg already
   at sigma=1e-4 and stays there (doesn't keep growing). Root cause: this
   is the mathematically NECESSARY discontinuity of any exact O(3)
   canonicalization exactly AT a symmetric configuration (already
   documented in this project's `wiki/notes/impossibility-continuous-
   canonicalization.md` and in `canonicalize_3d.py`'s own "assumption
   ledger": *"an adversarial cloud whose gap sits EXACTLY at the
   threshold flips regime across poses"*). It is not unique to ASUN when
   the TEST INPUT itself sits exactly on the discontinuity -- any exact
   method, ours included, has a jump there. The literal task-D "smoking
   gun" recipe (idealized benzene/methane at sigma=1e-3) therefore does
   NOT show ours beating ASUN -- we report this rather than pick a
   different molecule to make the number look better.
2. **On C3h/C4h specifically (a genuine >=3-fold axis, which the
   project's own theory says FORCES a repeated eigenvalue of X^T X)**,
   `ours` (the eigenframe-accelerated `canonicalize_3d`) also destabilizes
   almost immediately (qm-sym C3h: theta 44.7 deg already at sigma=1e-4;
   C4h: theta 42.9 deg at sigma=1e-4) -- because its regime dispatch is
   itself driven by the eigenvalue GAP, and a forced-degenerate spectrum
   sits exactly at that dispatch boundary. `ours_wwv` (PCA-free, no
   eigendecomposition) is measurably better on C3h (graceful: 0 -> 0.057 ->
   0.97 -> 11.9 -> 44.7 deg from sigma=0 to 1e-2) but not on C4h (also
   jumps immediately, 33.9 deg at sigma=1e-4) -- so the PCA-free
   construction reduces, but does not eliminate, sensitivity on the
   forced-degenerate strata. ASUN and PCA also degrade substantially here
   -- this looks like a genuinely hard regime for every tested method, not
   an ASUN-only weakness.
3. **Theta is separately ill-conditioned on exactly-linear (D*h/C*v)
   molecules**, independent of which method produced the poses: Procrustes
   fits a best rotation in 2 directions where a collinear cloud has zero
   signal, so it reports an essentially arbitrary angle (we measured
   theta=90 deg between two `ours` canonical poses of synthetic CO2 that
   agree to EMD~2e-16). For D*h/C*v strata, EMD is the metric to trust, not
   theta (documented in `metrics.py`).

Net assessment: the core claim holds, cleanly, on the strata that cover
the overwhelming majority of realistic near-symmetric molecular data
(sign-ambiguity-only symmetry: C1/Cs/C2/C2v/C2h, including QM-sym's
dominant C2h class) plus an unconditional crash-immunity advantage on
orthogonal-axis shapes (octahedron/bipyramids). On the rarer, genuinely
spectrally-degenerate strata (C3h/C4h/cubic/linear), all methods are
fragile near the exact symmetric point, for reasons intrinsic to exact
canonicalization (not an ASUN implementation defect) -- there, `ours`'s
real advantage over ASUN narrows to "never crashes" plus "EMD stays
valid," not "theta stays low."

### Smoking gun

**Real, near-symmetric data (the representative case: QM-sym molecule
`QM_sym_014009.xyz`, declared C2h, tiny sigma=1e-3):**

| method | sigma | EMD | theta (deg) | flip_rate |
|---|---|---|---|---|
| asun | 0.001 | 0.313 | 3.61  | 0.33 |
| ours | 0.001 | 0.0016 | 0.012 | 0.00 |

ASUN's frame-residual angle is ~300x ours's at the same tiny noise level,
on data whose designed symmetry (per QM-sym's own construction) does not
force a repeated eigenvalue. By sigma=0.01: ASUN theta=32.3 deg
(flip_rate=0.73) vs ours theta=0.12 deg (flip_rate=0.0).

**Literal task recipe (idealized exact molecule, sigma=1e-3) -- reported
for completeness, does not show the hoped effect, see complication #1
above:**

| method | molecule | sigma | EMD | theta (deg) | flip_rate |
|---|---|---|---|---|---|
| asun | methane (Td) | 0.001 | 1.2e-3 | 0.046 | 0.00 |
| ours | methane (Td) | 0.001 | 0.611  | 53.26 | 1.00 |
| asun | benzene (D6h) | 0.001 | 0.054 | 42.03 | 0.47 |
| ours | benzene (D6h) | 0.001 | 1.865 | 91.30 | 1.00 |

## Known caveats

- **QM9 census mismatch for C*v.** The background census quoted in the
  task brief (C*v ~3.2%) does not match what our pymatgen(tol=0.1) run
  finds in a random 3000-8000 molecule scan (C*v ~0.01-0.03%, i.e. ~1
  molecule). This is very likely because pymatgen's *separate*
  `eigen_tolerance` parameter (distinct from the `tolerance=0.1` distance
  tolerance we were told to fix) governs linear-molecule detection, and
  real DFT-relaxed near-linear QM9 geometries are evidently not close
  enough to exactly collinear for the default `eigen_tolerance` to call
  them C*v -- most get labeled Cs/C1 instead. We did NOT retune any
  pymatgen tolerance (per instructions, tol=0.1 fixed); we note the
  discrepancy rather than silently reconciling it. It does not affect the
  experiment's validity: the synthetic battery's CO2/acetylene give clean,
  exactly-linear D*h representatives regardless of how many QM9 molecules
  get the C*v label.
- **rdkit/QM9 incompatibility, worked around.** gnnplus's rdkit
  (2026.03.1) fails to parse some `gdb9.sdf` entries under
  torch_geometric 2.3.1's `QM9.process()` (mol=None from stricter modern
  sanitization). `load_qm9()` forces PyG's rdkit-unavailable code path,
  which loads the already-present `raw/qm9_v3.pt` pre-processed tensor
  cache instead -- identical `pos`/`z` fields, the only ones this harness
  uses.
- **QM-sym download succeeded** (see above) -- better than the
  "opportunistic, skip on failure" fallback the task allowed for, so no
  substitution was needed; the synthetic battery remains as designed
  regardless, for the exactly-idealized end of the spectrum.
