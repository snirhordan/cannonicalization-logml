# Results reproduction — "Hidden in Plain Symmetry"

This folder contains all code, vendored dependencies, and result CSVs that
recreate the two results reported on the project's results page
([`results.html`](results.html), included here for reference):

1. **ASUN robustness reproduction** (§2–§5): re-running Baker et al. 2024's own
   evaluation harness shows their published ASUN canonicalization is **not**
   invariant to rotation + translation on symmetric molecules, while our
   finite-group canonicalization is exact to machine precision.
2. **PCA-degenerate canonicalization construction** (§6–§8): an explicit,
   verified canonicalization for the one stratum PCA cannot resolve — a
   \(\ge 3\)-fold axis, which forces a double eigenvalue of \(X^\top X\) and
   leaves an \(O(2)\times O(1)\) residual.

Nothing here has been changed from the code that produced the page except that
the hard-coded absolute import paths were made relative (resolved from each
script's own location) so the folder is self-contained, and the QM9 dataset
root is now read from the `QM9_ROOT` environment variable (falling back to the
original default). No algorithm, tolerance, seed, or metric was touched.

## Layout

```
results_reproduction/
├── results.html                     # the page these scripts reproduce
├── code/                            # OUR canonicalizations + the degenerate construction
│   ├── canonicalize_3d.py           #   "ours" used by the ASUN harness (geometry-only)
│   ├── canonicalize_2d.py           #   2D polar routine
│   ├── canonicalize_pca_degenerate.py   # §6–§8 construction (O(2)×O(1) residual)
│   ├── canonicalize_pca_full.py     #   regime-dispatch full method (R0/DISTINCT/AXIAL/TRIPLE)
│   ├── test_canonicalize_pca_degenerate.py   # 28 tests — §8 verification table
│   ├── test_canonicalize_3d.py      #   12 tests
│   ├── test_canonicalize_2d.py      #   13 tests
│   └── other_code/sign_invariance.py     # canonical_pca / least-rotation helper
└── experiments/asun_robustness/     # the ASUN reproduction harness
    ├── reproduce_repo_tables.py     # Table 1 (by rank), serial
    ├── reproduce_repo_tables_mp.py  # Tables 1 & 2, multiprocessing (subset study + full QM9)
    ├── run_ae_table2.py             # AE (Winter et al. giae) column of Table 2
    ├── run_pca_full_dispatch.py     # regime-dispatch method over full QM9
    ├── run_sweep.py, methods.py, metrics.py, data_prep.py   # added-noise sweep + shared utils
    ├── gap_rtol_sweep/              # §5 honest caveat: added-noise robustness
    ├── submit_*.sbatch              # SLURM scripts (relative cd; run the commands below)
    ├── vendor/pyorbit/              # vendored ASUN (Utah-Math-Data-Science/alignment @ 17d2c7f)
    ├── vendor/giae_models/          # vendored Winter et al. autoencoder weights (giae)
    └── results/                     # the CSVs / JSON behind results.html (see mapping)
```

## Environments

- **`code/` tests** — pure NumPy. Any Python 3 with NumPy; the page ran them
  with `conda run -n base python3`.
- **ASUN harness** — needs `torch`, `torch_geometric` (QM9), `POT` (`ot`),
  `pointgroup`, `scipy`, and (for the AE column) `giae`. The page ran these in
  the `gnnplus` env (torch 2.2 / cu118). Install ASUN's extras if missing:
  `conda run -n gnnplus pip install pot pointgroup giae`.
- **Dataset** — QM9 via PyG. Set `QM9_ROOT` to your PyG QM9 root, or pass
  `--qm9root`; PyG downloads it on first use if absent. Default:
  `/home/snirhordan/canonicalization/data/pyg_qm9`.

> `conda run` treats `--n` as ambiguous with its own `--name`. Either call the
> env interpreter directly (`$(conda run -n gnnplus which python3) reproduce_repo_tables.py --n ...`)
> or use the provided `sbatch` scripts, which do exactly that.

## Reproducing each part of `results.html`

### §8 — PCA-degenerate construction verification (28/28)

```bash
cd code
conda run -n base python3 test_canonicalize_pca_degenerate.py     # -> 28/28 tests passed
conda run -n base python3 test_canonicalize_3d.py                 # -> 12/12
conda run -n base python3 test_canonicalize_2d.py                 # -> 13/13
```

This is the verification table in §8: source invariance, eigenframe
\(O(2)\times O(1)\) invariance, the staggered-ring trap (module holds; the
naïve 2-plane baseline fails), the axis-sign tie, and off-stratum refusal.

### §2 — Table 1 (by rank), `qm9[:10000]`, 10 perturbations

```bash
cd experiments/asun_robustness
PY=$(conda run -n gnnplus which python3)
$PY reproduce_repo_tables.py --n 10000 --nperturb 10 \
    --out results/reproduce_tables_10k.csv
```
Output → `results/reproduce_tables_10k.csv` (bundled), run log →
`results/reproduce_10k.log`. Reproduces the ASUN / PCA(centered) / ours rows;
the centered-PCA numbers match the paper's reported PCA row to ~4 sig figs,
validating the harness, while ASUN is nonzero on rank 3 and ours is exactly 0.

### §3 — Subset study, 5,000 molecules × 100 perturbations

```bash
cd experiments/asun_robustness
sbatch submit_repro5k.sbatch     # qm9[:5000]  -> results/reproduce_tables_paper5k100.csv
sbatch submit_subsets.sbatch     # random seeds 1..5 -> results/reproduce_rand5k_s{1..5}.csv
```
Both call `reproduce_repo_tables_mp.py --n 5000 --nperturb 100` (the random runs
add `--subset-seed S`). Shows the size-ordered `qm9[:5000]` slice (0.066) vs
random subsets (≈0.040) vs the paper's 0.028; ours stays at machine precision.

### §4 — Table 2 (by point group), full QM9 (130,831 molecules)

```bash
cd experiments/asun_robustness
sbatch submit_fullqm9.sbatch     # PCA(centered)/ASUN/ours -> results/table2_fullqm9.csv
sbatch submit_ae.sbatch          # AE (giae) column        -> results/ae_results.json
# optional: the regime-dispatch full method over QM9
sbatch submit_pca_full_dispatch.sbatch   # -> results/pca_full_dispatch.csv
```
`submit_fullqm9.sbatch` runs `reproduce_repo_tables_mp.py --n 130831
--nperturb 10`; `submit_ae.sbatch` runs `run_ae_table2.py` (300-molecule cap on
C1/Cs, all found for the rare groups), reproducing the paper's AE row including
its poor C∞v generalization.

### §5 — honest caveat: added-noise robustness

```bash
cd experiments/asun_robustness
sbatch submit_full.sbatch                              # full sigma sweep -> results/full.csv (gitignored, large)
conda run -n gnnplus python3 gap_rtol_sweep/noise_robustness_test.py   # gap_rtol degeneracy sweep
```
Under *added coordinate noise* (as opposed to the zero-noise rigid transform of
§2–§4), our method also destabilizes on genuine \(\ge 3\)-fold axes — the
degeneracy dichotomy this project is built around. `run_sweep.py` and
`gap_rtol_sweep/` quantify it. A pilot output is bundled as `results/pilot.csv`
/ `results/pilot_table.txt`.

## Bundled result artifacts (`experiments/asun_robustness/results/`)

| File | Reproduces |
|---|---|
| `reproduce_tables_10k.csv` | §2 Table 1 (by rank), qm9[:10000] |
| `reproduce_tables_paper5k100.csv` | §3 qm9[:5000] × 100 |
| `reproduce_rand5k_s{1..5}.csv` | §3 random 5000 subsets, seeds 1–5 |
| `table2_fullqm9.csv` | §4 PCA(centered)/ASUN/ours over full QM9 |
| `ae_results.json` | §4 AE (giae) column |
| `pilot.csv`, `pilot_table.txt`, `molecules_pilot.npz` | §5 added-noise pilot |
| `qm9_point_group_labels.csv` | pymatgen point-group cache |
| `slurm_*.out` | run logs for the SLURM jobs above |

The full-QM9 sigma sweep outputs (`full.csv`, `full_raw_per_molecule.csv`,
`molecules_full.npz`) are intentionally git-ignored (large; regenerated by
`submit_full.sbatch`).

## Provenance

- **ASUN** is vendored under `vendor/pyorbit/` from
  `github.com/Utah-Math-Data-Science/alignment` (commit `17d2c7f`, 2024-07-25),
  used verbatim — its `CatFrame` with the shipped `tol=1e-2`, never tuned.
- **AE** uses the Winter et al. learned autoencoder frame via `giae`, weights
  vendored under `vendor/giae_models/`.
- All metrics use the paper's own geometric EMD (POT `ot.emd2`, uniform
  weights, Euclidean cost); point groups via the `pointgroup` library; seed 42
  throughout the zero-noise runs.
