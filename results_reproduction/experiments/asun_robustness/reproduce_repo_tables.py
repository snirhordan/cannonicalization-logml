"""Faithful reproduction of ASUN's own Table 1 (by rank) & Table 2 (by point
group) from examples/unique_canonical_representation.ipynb in the released repo
(github.com/Utah-Math-Data-Science/alignment).

We copy their EXACT protocol verbatim:
  - QM9(root=...)[:N], seed 42, per molecule: baseline normalization, then
    nperturb x (R.random() rotation, np.random.rand(3) translation), translation
    added THEN rotated, applied to the original data.pos each time.
  - EMD = ot.emd2(unif, unif, ot.dist(pc1,pc2,'euclidean'))  (their geometric EMD).
  - rank = torch.linalg.matrix_rank(data.pos); point group via the `pointgroup`
    PyPI lib PointGroup(positions, symbols).get_point_group(), fallback 'C1'.

Methods:
  asun       -- CatFrame(tol=1e-2).get_frame(pos, z)          (their CELL 9)
  pca_svd    -- SVDAlignment.svd_rotate (eig of X^T X, desc, no sign fix, no center)
                = the INTENDED PCA baseline (their CELL 5's svd_rotate method)
  pca_noop   -- SVDAlignment.__call__ as literally written in CELL 5 (a no-op stub:
                normalized == raw cloud) -- included to show the notebook cell as
                shipped does not actually align.
  ours       -- our canonicalize_3d(pos).points (geometry-only), scored under the
                SAME EMD, for a like-for-like number.

Every method is applied to the SAME per-molecule perturbations (seed 42 consumed
once per molecule in rotation-then-translation order, exactly as each of their
per-method cells consumes it), so this single loop reproduces each cell's numbers.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch
import ot
from scipy.spatial.transform import Rotation as R

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.normpath(os.path.join(_HERE, '..', '..', 'code'))
sys.path.insert(0, os.path.join(_HERE, 'vendor', 'pyorbit'))
sys.path.insert(0, _CODE)
from CategoricalPointCloud import CatFrame            # ASUN (their released code)
from canonicalize_3d import canonicalize_3d           # ours

ATOM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}


def emd(pc1, pc2):
    """Their compute_wasserstein_distance, verbatim."""
    if isinstance(pc1, torch.Tensor):
        pc1 = pc1.detach().cpu().numpy()
    if isinstance(pc2, torch.Tensor):
        pc2 = pc2.detach().cpu().numpy()
    n1, n2 = pc1.shape[0], pc2.shape[0]
    a, b = ot.unif(n1), ot.unif(n2)
    M = ot.dist(pc1, pc2, metric='euclidean')
    return float(ot.emd2(a, b, M))


def svd_rotate(pc):
    """SVDAlignment.svd_rotate, verbatim (no centering, no sign fix)."""
    C = pc.t() @ pc
    e, v = torch.linalg.eig(C)
    e = torch.view_as_real(e)
    v = v.real
    idx = e[:, 0].argsort(descending=True)
    v = v.t()[idx].t()
    return pc @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--nperturb', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--qm9root', default=os.environ.get('QM9_ROOT', '/home/snirhordan/canonicalization/data/pyg_qm9'))
    ap.add_argument('--out', default='results/reproduce_tables.csv')
    args = ap.parse_args()

    from torch_geometric.datasets import QM9
    from pointgroup import PointGroup
    qm9 = QM9(root=args.qm9root)

    frame = CatFrame(tol=1e-2)
    methods = ['asun', 'pca_centered', 'pca_svd', 'pca_noop', 'ours']

    # accumulators: acc[key][method] = list of per-perturbation losses
    from collections import defaultdict
    rank_acc = {r: {m: [] for m in methods} for r in (1, 2, 3)}
    pg_acc = defaultdict(lambda: {m: [] for m in methods})
    pg_of = {}   # point group per molecule (for reporting counts)
    crash = {m: 0 for m in methods}

    np.random.seed(args.seed)
    t0 = time.time()
    N = min(args.n, len(qm9))
    for idx in range(N):
        data = qm9[idx]
        pos = data.pos                       # (n,3) torch, uncentered
        z = data.z.numpy()
        rank = int(torch.linalg.matrix_rank(pos))
        rank = 1 if rank <= 1 else (2 if rank == 2 else 3)
        symbols = [ATOM[int(a)] for a in z]
        try:
            pg = PointGroup(pos.numpy(), symbols).get_point_group()
        except Exception:
            pg = 'C1'
        pg_of[idx] = pg

        # baseline normalizations (deterministic; consume no RNG)
        base = {}
        try:
            base['asun'] = np.asarray(frame.get_frame(pos, z)[0], float)
        except Exception:
            base['asun'] = None
        base['pca_svd'] = svd_rotate(pos).detach().numpy()
        base['pca_centered'] = svd_rotate(pos - pos.mean(dim=0)).detach().numpy()
        base['pca_noop'] = pos.detach().numpy()
        try:
            base['ours'] = canonicalize_3d(pos.numpy()).points
        except Exception:
            base['ours'] = None

        for _ in range(args.nperturb):
            Rm = R.random().as_matrix()                 # consumes global RNG
            t = np.random.rand(3)                         # consumes global RNG
            pert = (Rm @ (pos + torch.from_numpy(t)).numpy().T).T   # translate then rotate
            pert_t = torch.from_numpy(pert).to(pos.dtype)

            new = {}
            try:
                new['asun'] = np.asarray(frame.get_frame(pert_t, z)[0], float)
            except Exception:
                new['asun'] = None
            new['pca_svd'] = svd_rotate(pert_t).detach().numpy()
            new['pca_centered'] = svd_rotate(pert_t - pert_t.mean(dim=0)).detach().numpy()
            new['pca_noop'] = pert
            try:
                new['ours'] = canonicalize_3d(pert).points
            except Exception:
                new['ours'] = None

            for m in methods:
                if base[m] is None or new[m] is None:
                    crash[m] += 1
                    continue
                loss = emd(base[m], new[m])
                rank_acc[rank][m].append(loss)
                pg_acc[pg][m].append(loss)

        if (idx + 1) % 200 == 0:
            print(f'  {idx+1}/{N}  ({time.time()-t0:.0f}s)', flush=True)

    def cell(m, acc, stat):
        v = acc[m]
        return f'{stat(v):.5f}' if v else 'n/a'

    from collections import Counter
    pgcount = Counter(pg_of.values())

    for stat, sname in ((np.mean, 'mean'), (np.median, 'median')):
        print(f'\n===== TABLE 1 (by rank) — {sname} EMD =====')
        print(f'{"method":12s} {"rank1":>10s} {"rank2":>10s} {"rank3":>10s}')
        for m in methods:
            print(f'{m:12s} ' + ' '.join(f'{cell(m,rank_acc[r],stat):>10s}' for r in (1, 2, 3)))
    print('\nPaper Table 1 (mean): PCA 0.00014/0.01793/0.82758 | AE 1.15122/0.03754/0.03178 | ASUN 0.00014/0.00008/0.02826')

    for stat, sname in ((np.mean, 'mean'), (np.median, 'median')):
        print(f'\n===== TABLE 2 (by point group) — {sname} EMD =====')
        order = sorted(pg_acc.keys(), key=lambda k: -pgcount[k])
        print(f'{"pg":8s} {"n_mol":>6s} ' + ' '.join(f'{m:>12s}' for m in methods))
        for pg in order:
            print(f'{pg:8s} {pgcount[pg]:6d} ' + ' '.join(f'{cell(m,pg_acc[pg],stat):>12s}' for m in methods))
    print('\nPaper Table 2 (C1/Cinfv/Cs/D6h/Td) mean: PCA .853/.303/.679/.244/.721 | AE .041/.240/.037/.023/.028 | ASUN .018/.001/.004/.001/.001')
    print(f'\ncrashes: {crash}')
    print(f'total time {time.time()-t0:.0f}s for N={N}, nperturb={args.nperturb}')

    # save tidy CSV
    import csv
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slice_kind', 'slice', 'method', 'n_obs', 'mean_emd', 'median_emd', 'n_mol'])
        for r in (1, 2, 3):
            for m in methods:
                v = rank_acc[r][m]
                w.writerow(['rank', r, m, len(v), np.mean(v) if v else '', np.median(v) if v else '', ''])
        for pg in sorted(pg_acc.keys(), key=lambda k: -pgcount[k]):
            for m in methods:
                v = pg_acc[pg][m]
                w.writerow(['pointgroup', pg, m, len(v), np.mean(v) if v else '', np.median(v) if v else '', pgcount[pg]])
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
