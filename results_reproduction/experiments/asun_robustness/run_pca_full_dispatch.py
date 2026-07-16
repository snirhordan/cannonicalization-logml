"""run_pca_full_dispatch.py -- ONE method (canonicalize_pca_full, DEFAULT_GAP_RTOL=2e-2
after Phase 2 tuning -- the only constant edited in canonicalize_pca_full.py, raised from
the original DEFAULT_GAP_RTOL=1e-3) run over QM9, producing the exact two slices needed
for ONE new row in each of results.html's two tables:

  (a) per-RANK over qm9[:10000], 10 perturbations/molecule (Table 1's protocol).
  (b) per-POINT-GROUP over the FULL QM9 (130,831 molecules), 10 perturbations/molecule
      (Table 2's protocol).

Both slices come from ONE pass over the full 130,831-molecule dataset (nperturb=10, seed
42, RandomState(seed+idx) per molecule -- identical harness convention to
reproduce_repo_tables_mp.py) since qm9[:10000] is a prefix of the full set and uses the
SAME per-molecule-idx-seeded perturbations either way; the (a) aggregation just restricts
to idx < 10000, matching Table 1's protocol exactly without a second, redundant pass.

Perturbation: random rotation (Haar, scipy Rotation.random) + random translation
(np.random.rand(3)), NO coordinate noise -- identical to reproduce_repo_tables_mp.py.
Metric: the paper's own POT geometric EMD (ot.emd2, uniform weights, Euclidean cost).
Point-group labels: `pointgroup` library, on-the-fly per molecule (no cache), same
'C1' fallback on exception -- identical to reproduce_repo_tables_mp.py.

canonicalize_pca_full NEVER raises (see its module docstring): every molecule gets a
regime label (R0 / DISTINCT / AXIAL / TRIPLE) and a canonical pose. The dispatch decision
(the two eigenvalue-gap tests) is itself a function of the ROTATION-INVARIANT eigenvalues
of X^T X, so baseline and perturbed copies of the same molecule should almost always land
in the SAME regime and give EMD ~ machine precision; the only place a nonzero EMD tail can
appear is a molecule sitting close enough to the gap_rtol=2e-2 boundary that floating-point
roundoff in the perturbed copy's eigenvalues flips it across the DISTINCT/AXIAL or
AXIAL/TRIPLE boundary into a *different* handler than the baseline used. We track this
directly per molecule (base regime vs. each perturbation's regime, plus max EMD across its
10 perturbations) so the report can name that tail rather than average over it silently.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import ot
from scipy.spatial.transform import Rotation as R

_CODE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
sys.path.insert(0, _CODE)
from canonicalize_pca_full import canonicalize_pca_full, DEFAULT_GAP_RTOL  # noqa: E402

ATOM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def emd(pc1: np.ndarray, pc2: np.ndarray) -> float:
    a, b = ot.unif(pc1.shape[0]), ot.unif(pc2.shape[0])
    M = ot.dist(pc1, pc2, metric="euclidean")
    return float(ot.emd2(a, b, M))


def process_molecule(task):
    """Worker: one molecule. task = (idx, pos(np f64), z(np int), nperturb, seed)."""
    idx, pos, z, nperturb, seed = task
    rank = int(np.linalg.matrix_rank(pos))
    rank = 1 if rank <= 1 else (2 if rank == 2 else 3)
    symbols = [ATOM[int(a)] for a in z]
    try:
        pg = __import__("pointgroup", fromlist=["PointGroup"]).PointGroup(pos, symbols).get_point_group()
    except Exception:
        pg = "C1"

    base = canonicalize_pca_full(pos)
    losses = []
    mismatches = 0
    rng = np.random.RandomState(seed + idx)  # per-molecule, reproducible + independent
    for _ in range(nperturb):
        Rm = R.random(random_state=rng).as_matrix()
        t = rng.rand(3)
        pert = (Rm @ (pos + t).T).T
        new = canonicalize_pca_full(pert)
        losses.append(emd(base.points, new.points))
        if new.regime != base.regime:
            mismatches += 1
    return {
        "idx": idx,
        "rank": rank,
        "pg": pg,
        "losses": losses,
        "base_regime": base.regime,
        "mismatches": mismatches,
        "max_loss": max(losses) if losses else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rank", type=int, default=10000,
                    help="rank slice size, matching Table 1's qm9[:N] protocol")
    ap.add_argument("--n-full", type=int, default=130831,
                    help="point-group slice size (full QM9), matching Table 2's protocol")
    ap.add_argument("--nperturb", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--qm9root", default=os.environ.get("QM9_ROOT", "/home/snirhordan/canonicalization/data/pyg_qm9"))
    ap.add_argument("--out", default="results/pca_full_dispatch.csv")
    args = ap.parse_args()

    from torch_geometric.datasets import QM9
    qm9 = QM9(root=args.qm9root)
    N = min(args.n_full, len(qm9))
    print(f"canonicalize_pca_full (DEFAULT_GAP_RTOL={DEFAULT_GAP_RTOL}) over {N} QM9 "
          f"molecules, {args.nperturb} perturbations each; rank-slice = first "
          f"{args.n_rank}, point-group-slice = full {N}...", flush=True)

    tasks = []
    for i in range(N):
        d = qm9[i]
        tasks.append((i, d.pos.numpy().astype(np.float64), d.z.numpy(), args.nperturb, args.seed))

    t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(args.cores) as pool:
        results = []
        for k, r in enumerate(pool.imap_unordered(process_molecule, tasks, chunksize=32)):
            results.append(r)
            if (k + 1) % 10000 == 0:
                print(f"  {k+1}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    rank_acc = {rk: [] for rk in (1, 2, 3)}
    pg_acc = defaultdict(list)
    pgcount = Counter()
    regime_count = Counter()
    boundary_mols = []  # (idx, base_regime, mismatches, max_loss)

    for r in results:
        pgcount[r["pg"]] += 1
        pg_acc[r["pg"]].extend(r["losses"])
        regime_count[r["base_regime"]] += 1
        if r["idx"] < args.n_rank:
            rank_acc[r["rank"]].extend(r["losses"])
        if r["mismatches"] > 0 or r["max_loss"] > 1e-6:
            boundary_mols.append((r["idx"], r["base_regime"], r["mismatches"], r["max_loss"]))

    print(f"\n===== per-RANK mean EMD (qm9[:{args.n_rank}], nperturb={args.nperturb}) =====")
    for rk in (1, 2, 3):
        v = rank_acc[rk]
        if v:
            print(f"  rank{rk}: n_obs={len(v)} mean={np.mean(v):.6e} median={np.median(v):.6e}")
        else:
            print(f"  rank{rk}: n/a (no molecules)")

    print(f"\n===== per-POINT-GROUP mean EMD (full QM9 N={N}, nperturb={args.nperturb}) =====")
    order = sorted(pg_acc.keys(), key=lambda k: -pgcount[k])
    for pg in order:
        v = pg_acc[pg]
        print(f"  {pg:8s} n_mol={pgcount[pg]:6d} n_obs={len(v)} mean={np.mean(v):.6e} median={np.median(v):.6e}")

    print(f"\n===== dispatch regime counts (base pose, {N} molecules) =====")
    for reg in ("R0", "DISTINCT", "AXIAL", "TRIPLE"):
        print(f"  {reg:9s} {regime_count.get(reg, 0)}")

    print(f"\n===== boundary molecules (regime mismatch across a perturbation, or "
          f"max EMD > 1e-6 across its {args.nperturb} perturbations): {len(boundary_mols)} "
          f"/ {N} =====")
    for idx, reg, mm, ml in sorted(boundary_mols, key=lambda x: -x[3])[:25]:
        print(f"  idx={idx} base_regime={reg} mismatches={mm}/{args.nperturb} max_loss={ml:.3e}")

    print(f"\ntotal time {time.time()-t0:.0f}s for N={N}, nperturb={args.nperturb}, cores={args.cores}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice_kind", "slice", "method", "n_obs", "mean_emd", "median_emd", "n_mol"])
        for rk in (1, 2, 3):
            v = rank_acc[rk]
            w.writerow(["rank", rk, "pca_full_dispatch", len(v),
                        np.mean(v) if v else "", np.median(v) if v else "", ""])
        for pg in order:
            v = pg_acc[pg]
            w.writerow(["pointgroup", pg, "pca_full_dispatch", len(v),
                        np.mean(v) if v else "", np.median(v) if v else "", pgcount[pg]])
        for reg in ("R0", "DISTINCT", "AXIAL", "TRIPLE"):
            w.writerow(["regime_count", reg, "pca_full_dispatch", regime_count.get(reg, 0), "", "", ""])
        w.writerow(["boundary_mol_count", "", "pca_full_dispatch", len(boundary_mols), "", "", N])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
