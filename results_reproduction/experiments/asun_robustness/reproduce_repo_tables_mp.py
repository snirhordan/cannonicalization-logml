"""Parallel (multiprocessing) version of reproduce_repo_tables.py.

Same faithful ASUN harness (their POT geometric EMD, `pointgroup` labels,
translation+rotation perturbations, NO coordinate noise), but molecules are
processed independently across CPU cores. GPUs do not help this workload: the
cost is ASUN's convex-hull + DFA (scipy Qhull + Hopcroft, sequential/combinatorial)
and POT's LP-based EMD -- both CPU-only. The speedup is pure CPU parallelism.

Reproducibility under parallelism: each molecule draws its own perturbations from
an independent RandomState(seed + idx), so results are deterministic and
order-independent (the serial script's single global RNG cannot be parallelized).
Aggregate means/medians match the serial version within Monte-Carlo noise.
"""
from __future__ import annotations
import argparse, sys, time, os, csv
from collections import defaultdict, Counter
import numpy as np
import torch
import ot
from scipy.spatial.transform import Rotation as R

HARNESS = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.normpath(os.path.join(HARNESS, '..', '..', 'code'))
sys.path.insert(0, HARNESS + '/vendor/pyorbit')
sys.path.insert(0, _CODE)
from CategoricalPointCloud import CatFrame
from canonicalize_3d import canonicalize_3d

ATOM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
METHODS = ['asun', 'pca_centered', 'pca_svd', 'pca_noop', 'ours']


def emd(pc1, pc2):
    if isinstance(pc1, torch.Tensor):
        pc1 = pc1.detach().cpu().numpy()
    if isinstance(pc2, torch.Tensor):
        pc2 = pc2.detach().cpu().numpy()
    a, b = ot.unif(pc1.shape[0]), ot.unif(pc2.shape[0])
    M = ot.dist(pc1, pc2, metric='euclidean')
    return float(ot.emd2(a, b, M))


def svd_rotate_np(pc):
    """SVDAlignment.svd_rotate, verbatim, on a numpy (n,3) array."""
    t = torch.from_numpy(np.ascontiguousarray(pc))
    C = t.t() @ t
    e, v = torch.linalg.eig(C)
    e = torch.view_as_real(e)
    v = v.real
    idx = e[:, 0].argsort(descending=True)
    v = v.t()[idx].t()
    return (t @ v).numpy()


def poses(frame, pos, z):
    """Return dict method -> canonical pose (n,3) or None on crash."""
    out = {}
    try:
        out['asun'] = np.asarray(frame.get_frame(pos, z)[0], float)
    except Exception:
        out['asun'] = None
    out['pca_centered'] = svd_rotate_np(pos - pos.mean(0))
    out['pca_svd'] = svd_rotate_np(pos)
    out['pca_noop'] = pos
    try:
        out['ours'] = canonicalize_3d(pos).points
    except Exception:
        out['ours'] = None
    return out


def process_molecule(task):
    """Worker: one molecule. task = (idx, pos(np f64), z(np int), nperturb, seed)."""
    idx, pos, z, nperturb, seed = task
    frame = CatFrame(tol=1e-2)                       # cheap; per-call avoids shared state
    rank = int(np.linalg.matrix_rank(pos))
    rank = 1 if rank <= 1 else (2 if rank == 2 else 3)
    symbols = [ATOM[int(a)] for a in z]
    try:
        pg = __import__('pointgroup', fromlist=['PointGroup']).PointGroup(pos, symbols).get_point_group()
    except Exception:
        pg = 'C1'

    base = poses(frame, pos, z)
    losses = {m: [] for m in METHODS}
    crash = {m: 0 for m in METHODS}
    rng = np.random.RandomState(seed + idx)          # per-molecule, reproducible + independent
    for _ in range(nperturb):
        Rm = R.random(random_state=rng).as_matrix()
        t = rng.rand(3)
        pert = (Rm @ (pos + t).T).T
        new = poses(frame, pert, z)
        for m in METHODS:
            if base[m] is None or new[m] is None:
                crash[m] += 1
                continue
            losses[m].append(emd(base[m], new[m]))
    return {'idx': idx, 'rank': rank, 'pg': pg, 'losses': losses, 'crash': crash}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--nperturb', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cores', type=int, default=8)
    ap.add_argument('--subset-seed', type=int, default=-1,
                    help='<0: first N molecules (qm9[:N]); >=0: random N molecules sampled with this seed')
    ap.add_argument('--qm9root', default=os.environ.get('QM9_ROOT', '/home/snirhordan/canonicalization/data/pyg_qm9'))
    ap.add_argument('--out', default='results/reproduce_tables_mp.csv')
    args = ap.parse_args()

    from torch_geometric.datasets import QM9
    qm9 = QM9(root=args.qm9root)
    N = min(args.n, len(qm9))
    if args.subset_seed < 0:
        idxs = list(range(N))
        tag = f'first{N}'
    else:
        srng = np.random.RandomState(args.subset_seed)
        idxs = sorted(int(i) for i in srng.choice(len(qm9), size=N, replace=False))
        tag = f'rand{N}_s{args.subset_seed}'
    print(f'SUBSET={tag}  extracting {N} molecules (perturbation seed tied to original QM9 index)...', flush=True)
    tasks = []
    for i in idxs:
        d = qm9[i]
        tasks.append((i, d.pos.numpy().astype(np.float64), d.z.numpy(), args.nperturb, args.seed))

    t0 = time.time()
    import multiprocessing as mp
    print(f'processing {N} molecules x {args.nperturb} perturb on {args.cores} cores...', flush=True)
    with mp.Pool(args.cores) as pool:
        results = []
        for k, r in enumerate(pool.imap_unordered(process_molecule, tasks, chunksize=8)):
            results.append(r)
            if (k + 1) % 500 == 0:
                print(f'  {k+1}/{N}  ({time.time()-t0:.0f}s)', flush=True)

    rank_acc = {rk: {m: [] for m in METHODS} for rk in (1, 2, 3)}
    pg_acc = defaultdict(lambda: {m: [] for m in METHODS})
    pgcount = Counter()
    crash_tot = {m: 0 for m in METHODS}
    for r in results:
        pgcount[r['pg']] += 1
        for m in METHODS:
            rank_acc[r['rank']][m].extend(r['losses'][m])
            pg_acc[r['pg']][m].extend(r['losses'][m])
            crash_tot[m] += r['crash'][m]

    def cell(m, acc, stat):
        v = acc[m]
        return f'{stat(v):.5f}' if v else 'n/a'

    for stat, sname in ((np.mean, 'mean'), (np.median, 'median')):
        print(f'\n===== TABLE 1 (by rank) — {sname} EMD =====')
        print(f'{"method":12s} {"rank1":>10s} {"rank2":>10s} {"rank3":>10s}')
        for m in METHODS:
            print(f'{m:12s} ' + ' '.join(f'{cell(m,rank_acc[r],stat):>10s}' for r in (1, 2, 3)))
    print('\nPaper Table 1 (mean): PCA 0.00014/0.01793/0.82758 | AE 1.15122/0.03754/0.03178 | ASUN 0.00014/0.00008/0.02826')

    for stat, sname in ((np.mean, 'mean'), (np.median, 'median')):
        print(f'\n===== TABLE 2 (by point group) — {sname} EMD =====')
        order = sorted(pg_acc.keys(), key=lambda k: -pgcount[k])
        print(f'{"pg":8s} {"n_mol":>6s} ' + ' '.join(f'{m:>12s}' for m in METHODS))
        for pg in order:
            print(f'{pg:8s} {pgcount[pg]:6d} ' + ' '.join(f'{cell(m,pg_acc[pg],stat):>12s}' for m in METHODS))
    print('\nPaper Table 2 (C1/Cinfv/Cs/D6h/Td) mean: PCA .853/.303/.679/.244/.721 | AE .041/.240/.037/.023/.028 | ASUN .018/.001/.004/.001/.001')
    print(f'\ncrashes: {crash_tot}')
    print(f'total time {time.time()-t0:.0f}s for N={N}, nperturb={args.nperturb}, cores={args.cores}')
    r3 = rank_acc[3]
    n_r3_mol = sum(1 for r in results if r['rank'] == 3)
    print(f'SUMMARY subset={tag} n_rank3_mol={n_r3_mol} '
          f'ASUN_rank3_mean={np.mean(r3["asun"]):.5f} ASUN_rank3_median={np.median(r3["asun"]):.6f} '
          f'ours_rank3_mean={np.mean(r3["ours"]):.2e} pca_rank3_mean={np.mean(r3["pca_centered"]):.5f}', flush=True)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slice_kind', 'slice', 'method', 'n_obs', 'mean_emd', 'median_emd', 'n_mol'])
        for rk in (1, 2, 3):
            for m in METHODS:
                v = rank_acc[rk][m]
                w.writerow(['rank', rk, m, len(v), np.mean(v) if v else '', np.median(v) if v else '', ''])
        for pg in sorted(pg_acc.keys(), key=lambda k: -pgcount[k]):
            for m in METHODS:
                v = pg_acc[pg][m]
                w.writerow(['pointgroup', pg, m, len(v), np.mean(v) if v else '', np.median(v) if v else '', pgcount[pg]])
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
