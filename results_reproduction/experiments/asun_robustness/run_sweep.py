"""run_sweep.py -- CLI driver for the ASUN-robustness harness.

Usage:
  conda run -n gnnplus python3 run_sweep.py --mode pilot --cores 2  --out results/pilot.csv
  conda run -n gnnplus python3 run_sweep.py --mode full  --cores 64 --out results/full.csv

For every (molecule, perturbation index p) task:
  - A single per-(molecule, p) seed (independent of sigma) determines BOTH a
    Haar-random rotation Q_p (scipy Rotation.random) AND a unit Gaussian
    noise matrix Z_p ~ Normal(0, 1) of shape (n, 3), drawn from the SAME
    numpy RandomState in that order. For each sigma in --sigmas, the actual
    noise used is sigma * Z_p -- i.e. noise direction is fixed per (molecule,
    p) and only its MAGNITUDE grows with sigma. This is deliberate: it
    isolates "how does instability grow with noise scale" from "did we
    happen to sample a different adversarial direction at each sigma," and
    it makes sigma=0 an exact no-noise case (sigma * Z_p = 0) rather than a
    separate code path.
  - baseline  = molecule.coords                          (clean, centered)
  - perturbed = molecule.coords @ Q_p.T + sigma * Z_p     (rotated, noised)
  - ALL FOUR METHODS see the identical Q_p, Z_p for this (molecule, p) --
    the seed is a pure function of (molecule.name, p, --seed), recomputed
    independently inside each worker, so this holds regardless of
    multiprocessing scheduling order.
  - Because every method is a deterministic pure function of its input, the
    canonical pose of the CLEAN baseline does not depend on p or sigma, so
    it is computed ONCE per (molecule, method) in the main process before
    the pool starts (not recomputed on every task) -- same semantics as
    computing it fresh per task, far less redundant work over 100
    perturbations.

Outputs two CSVs (both written for --out=PATH.csv):
  PATH.csv                       tidy aggregate, one row per
                                  (method, dataset, point_group, sigma),
                                  pooling over ALL molecules and
                                  perturbations in that stratum.
  PATH_raw_per_molecule.csv      one row per (method, dataset, point_group,
                                  molecule, sigma), pooling over
                                  perturbations only -- for later
                                  figure-making / per-molecule drill-down.
Both report crash_rate, mean/median EMD, mean/median frame-residual theta
(deg), flip_rate (theta > 5 deg), and n (with n_crashed broken out) -- crashes
are NEVER folded into the EMD/theta means as 0 or silently dropped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_prep import (  # noqa: E402
    MoleculeRecord, build_synthetic_battery, load_qm9, label_qm9_indices,
    build_stratified_qm9_sample, try_qm_sym_download, build_qm_sym_sample,
    save_molecules, load_molecules, census,
)
from methods import METHODS, MethodResult  # noqa: E402
from metrics import consistency_metrics, is_flip  # noqa: E402

DEFAULT_SIGMAS = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
QM_SYM_DIR = os.path.join(_HERE, "data", "qm_sym_raw")


def _seed_for(mol_name: str, p: int, base_seed: int) -> int:
    """Deterministic 32-bit seed, stable across processes/runs (unlike
    Python's randomized str hash())."""
    s = f"{mol_name}|{p}|{base_seed}".encode()
    return zlib.crc32(s) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Molecule-set construction
# ---------------------------------------------------------------------------

def build_molecule_set(mode: str, seed: int, cores: int,
                        qm9_pilot_scan: int = 3000) -> List[MoleculeRecord]:
    rng = np.random.RandomState(seed)

    print("[run_sweep] building synthetic battery...")
    synth = build_synthetic_battery()

    ds = load_qm9()
    if mode == "pilot":
        idxs = rng.choice(len(ds), size=min(qm9_pilot_scan, len(ds)), replace=False).tolist()
        labels = label_qm9_indices(ds, idxs, n_workers=min(cores, 4))
        strata = sorted({pg for pg in labels.values() if not pg.startswith("ERROR:")})
        caps = {pg: 8 for pg in strata}  # "~8 molecules per stratum" (task spec, pilot mode)
        qm9_sample = build_stratified_qm9_sample(ds, labels, caps=caps, rng=rng)
        qm_sym_per_tar = 3
    else:
        idxs = list(range(len(ds)))
        labels = label_qm9_indices(ds, idxs, n_workers=cores)
        qm9_sample = build_stratified_qm9_sample(ds, labels, caps=None, rng=rng)  # DEFAULT_CAPS
        qm_sym_per_tar = 500

    c = census(labels)
    print(f"[run_sweep] QM9 census over {len(idxs)} scanned molecules: "
          + ", ".join(f"{pg}={n}" for pg, n in sorted(c.items(), key=lambda kv: -kv[1])))
    print(f"[run_sweep] QM9 stratified sample: {len(qm9_sample)} molecules")

    qm_sym_records: List[MoleculeRecord] = []
    cached_zip = os.path.join(QM_SYM_DIR, "1")
    zip_path = cached_zip if os.path.exists(cached_zip) else try_qm_sym_download(QM_SYM_DIR, timeout_s=60)
    if zip_path is not None:
        try:
            qm_sym_records = build_qm_sym_sample(zip_path, per_tar=qm_sym_per_tar, rng=rng)
        except Exception as e:  # noqa: BLE001
            print(f"[run_sweep] QM-sym extraction failed ({type(e).__name__}: {e}); "
                  "SUBSTITUTION: proceeding without QM-sym.")
    else:
        print("[run_sweep] QM-sym not available -- SUBSTITUTION: relying on the synthetic "
              "battery's C2h/C3h/C4h-adjacent shapes for the heavy-symmetry regime.")

    molecules = synth + qm9_sample + qm_sym_records
    print(f"[run_sweep] total molecule set: {len(molecules)} "
          f"(synthetic={len(synth)}, qm9={len(qm9_sample)}, qm-sym={len(qm_sym_records)})")
    return molecules


# ---------------------------------------------------------------------------
# Worker-side globals (populated before Pool() so fork gives COW sharing)
# ---------------------------------------------------------------------------

_MOLECULES: List[MoleculeRecord] = []
_BASELINE = {}          # (mol_idx, method_name) -> MethodResult
_SIGMAS: List[float] = []
_SEED_BASE: int = 0


def _init_worker_globals(molecules, baseline, sigmas, seed_base):
    global _MOLECULES, _BASELINE, _SIGMAS, _SEED_BASE
    _MOLECULES, _BASELINE, _SIGMAS, _SEED_BASE = molecules, baseline, sigmas, seed_base


def _compute_baseline(molecules: List[MoleculeRecord]):
    """Canonicalize the clean pose once per (molecule, method); deterministic
    methods make this exactly equivalent to recomputing it inside every
    (molecule, p, sigma) task, at a fraction of the cost."""
    baseline = {}
    t0 = time.time()
    for mi, mol in enumerate(molecules):
        for name, fn in METHODS.items():
            baseline[(mi, name)] = fn(mol.coords.copy(), mol.species.copy())
    n_crashed = sum(1 for v in baseline.values() if v.crashed)
    print(f"[run_sweep] baseline canonicalization: {len(molecules)} molecules x "
          f"{len(METHODS)} methods in {time.time()-t0:.1f}s "
          f"({n_crashed} baseline crashes out of {len(baseline)})")
    return baseline


def _task(args):
    mol_idx, p = args
    mol = _MOLECULES[mol_idx]
    seed = _seed_for(mol.name, p, _SEED_BASE)
    rng = np.random.RandomState(seed)
    Q = Rotation.random(random_state=rng).as_matrix()
    Z = rng.standard_normal(size=mol.coords.shape)

    rows = []
    for sigma in _SIGMAS:
        perturbed = mol.coords @ Q.T + sigma * Z
        for name, fn in METHODS.items():
            base_res: MethodResult = _BASELINE[(mol_idx, name)]
            pert_res: MethodResult = fn(perturbed.copy(), mol.species.copy())
            crashed = bool(base_res.crashed or pert_res.crashed)
            row = dict(
                dataset=mol.dataset, point_group=mol.point_group, molecule=mol.name,
                method=name, sigma=sigma, p=p, crashed=crashed,
                err_type_base=base_res.err_type, err_type_pert=pert_res.err_type,
                emd=np.nan, theta_deg=np.nan, flip=np.nan,
            )
            if not crashed:
                cm = consistency_metrics(base_res.pose, base_res.species_out,
                                          pert_res.pose, pert_res.species_out)
                row["emd"] = cm.emd
                row["theta_deg"] = cm.theta_deg
                row["flip"] = bool(is_flip(cm.theta_deg))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_block(g: pd.DataFrame) -> pd.Series:
    crashed = g["crashed"].astype(bool)
    valid = g.loc[~crashed]
    return pd.Series({
        "n_total": len(g),
        "n_crashed": int(crashed.sum()),
        "n_valid": len(valid),
        "crash_rate": float(crashed.mean()),
        "mean_emd": float(valid["emd"].mean()) if len(valid) else np.nan,
        "median_emd": float(valid["emd"].median()) if len(valid) else np.nan,
        "mean_theta_deg": float(valid["theta_deg"].mean()) if len(valid) else np.nan,
        "median_theta_deg": float(valid["theta_deg"].median()) if len(valid) else np.nan,
        "flip_rate": float(valid["flip"].mean()) if len(valid) else np.nan,
    })


def aggregate_raw_per_molecule(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["method", "dataset", "point_group", "molecule", "sigma"])
              .apply(_agg_block, include_groups=False).reset_index())


def aggregate_tidy(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby(["method", "dataset", "point_group", "sigma"])
             .apply(_agg_block, include_groups=False).reset_index())
    n_mol = (df.groupby(["method", "dataset", "point_group", "sigma"])["molecule"]
               .nunique().rename("n_molecules").reset_index())
    return out.merge(n_mol, on=["method", "dataset", "point_group", "sigma"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--out", required=True, help="tidy aggregate CSV path; raw CSV is derived from it")
    ap.add_argument("--sigmas", type=str, default=",".join(str(s) for s in DEFAULT_SIGMAS))
    ap.add_argument("--nperturb", type=int, default=None,
                     help="default: 15 (pilot) or 100 (full), per task spec")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qm9-pilot-scan", type=int, default=3000)
    ap.add_argument("--molecules-npz", type=str, default=None,
                     help="reuse a previously-saved molecule set instead of rebuilding it")
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",") if s != ""]
    nperturb = args.nperturb if args.nperturb is not None else (15 if args.mode == "pilot" else 100)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.molecules_npz and os.path.exists(args.molecules_npz):
        print(f"[run_sweep] loading cached molecule set from {args.molecules_npz}")
        molecules = load_molecules(args.molecules_npz)
    else:
        molecules = build_molecule_set(args.mode, args.seed, args.cores,
                                        qm9_pilot_scan=args.qm9_pilot_scan)
        mol_cache_path = args.molecules_npz or os.path.join(
            os.path.dirname(os.path.abspath(args.out)), f"molecules_{args.mode}.npz")
        save_molecules(molecules, mol_cache_path)
        print(f"[run_sweep] saved molecule set to {mol_cache_path}")

    # per-stratum n, so thin strata are visible (fairness requirement)
    strat_n = {}
    for m in molecules:
        strat_n[(m.dataset, m.point_group)] = strat_n.get((m.dataset, m.point_group), 0) + 1
    print("[run_sweep] per-stratum molecule counts:")
    for (ds, pg), n in sorted(strat_n.items()):
        print(f"    {ds:10s} {pg:8s} n={n}")

    baseline = _compute_baseline(molecules)

    tasks = [(mi, p) for mi in range(len(molecules)) for p in range(nperturb)]
    print(f"[run_sweep] {len(tasks)} (molecule, perturbation) tasks "
          f"({len(molecules)} molecules x {nperturb} perturbations), "
          f"sigmas={sigmas}, cores={args.cores}")

    t0 = time.time()
    if args.cores <= 1:
        _init_worker_globals(molecules, baseline, sigmas, args.seed)
        all_rows = [row for t in tasks for row in _task(t)]
    else:
        with Pool(processes=args.cores, initializer=_init_worker_globals,
                  initargs=(molecules, baseline, sigmas, args.seed)) as pool:
            chunks = pool.map(_task, tasks, chunksize=max(1, len(tasks) // (args.cores * 8) or 1))
        all_rows = [row for chunk in chunks for row in chunk]
    print(f"[run_sweep] sweep finished in {time.time()-t0:.1f}s, {len(all_rows)} rows")

    df = pd.DataFrame(all_rows)
    assert not np.isinf(df["emd"].fillna(0)).any(), "inf leaked into EMD"
    assert not np.isinf(df["theta_deg"].fillna(0)).any(), "inf leaked into theta"

    tidy = aggregate_tidy(df)
    raw = aggregate_raw_per_molecule(df)

    tidy.to_csv(args.out, index=False)
    raw_path = os.path.splitext(args.out)[0] + "_raw_per_molecule.csv"
    raw.to_csv(raw_path, index=False)
    print(f"[run_sweep] wrote tidy aggregate -> {args.out} ({len(tidy)} rows)")
    print(f"[run_sweep] wrote raw per-molecule -> {raw_path} ({len(raw)} rows)")

    print("\n[run_sweep] GATE check (sigma=0, all methods should be ~consistent):")
    z = tidy[tidy["sigma"] == 0.0]
    print(z[["method", "dataset", "point_group", "n_molecules", "crash_rate",
             "mean_emd", "mean_theta_deg", "flip_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
