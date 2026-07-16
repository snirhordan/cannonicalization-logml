"""AE (Winter et al. learned frame, via giae) column of the paper's Table 2.

Runs the giae autoencoder normalization exactly as in the ASUN repo's
unique_canonical_representation.ipynb cell 7, over QM9, for the paper's five
Table-2 point groups. Perturbation = random rotation + translation (NO noise);
EMD = their POT geometric EMD. Writes results/ae_results.json.

Model files are vendored under vendor/giae_models/ so this does not depend on
any ephemeral tmp. giae must be installed in the env (pip install
git+https://github.com/Bayer-Group/giae).
"""
import sys, time, random, json, os
import numpy as np
import torch

HARNESS = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HARNESS, 'vendor', 'giae_models')
sys.path.insert(0, MODELS)
from giae_model import Model
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from pointgroup import PointGroup
from scipy.spatial.transform import Rotation as R
import ot
from collections import defaultdict, Counter

random.seed(0); np.random.seed(0); torch.manual_seed(0)
Z2SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
TARGET = {'C1', 'Cinfv', 'Cs', 'D6h', 'Td'}
CAP = {'C1': 300, 'Cs': 300, 'Cinfv': 10**9, 'D6h': 10**9, 'Td': 10**9}
NPERTURB = 10


def main():
    model = Model(hidden_dim=256, emb_dim=32, num_layers=5).to('cpu')
    model.load_state_dict(torch.load(os.path.join(MODELS, 'giae_model.pth'), map_location='cpu'))
    model.eval()
    ds = QM9(root=os.environ.get('QM9_ROOT', '/home/snirhordan/canonicalization/data/pyg_qm9'))
    n_total = len(ds)
    print('QM9 total:', n_total, flush=True)

    t0 = time.time()
    idx_by_group = defaultdict(list)
    counts = Counter()
    for i in range(n_total):
        data = ds[i]
        symbols = [Z2SYM[int(z)] for z in data.z.numpy()]
        try:
            pg = PointGroup(data.pos.numpy(), symbols).get_point_group()
        except Exception:
            continue
        counts[pg] += 1
        if pg in TARGET:
            idx_by_group[pg].append(i)
        if (i + 1) % 20000 == 0:
            print(f'  scanned {i+1}/{n_total} ({time.time()-t0:.0f}s)', flush=True)
    print(f'scan done {time.time()-t0:.0f}s; target counts: '
          + ', '.join(f'{g}={len(idx_by_group[g])}' for g in TARGET), flush=True)

    sample = {}
    for g in TARGET:
        pool = idx_by_group[g]
        sample[g] = random.sample(pool, CAP[g]) if len(pool) > CAP[g] else list(pool)

    def normalize(batch):
        with torch.no_grad():
            pos_out, perm, vout, rot = model(batch, hard=False)
        pos_out = pos_out - pos_out.mean(0)
        return (pos_out @ rot.squeeze()).detach().numpy()

    results = {g: [] for g in TARGET}
    t2 = time.time(); n_done = 0; n_fail = 0
    for g in TARGET:
        for idx in sample[g]:
            data = ds[idx]
            batch = next(iter(DataLoader([data], batch_size=1)))
            batch.pos = batch.pos - batch.pos.mean(0)
            orig = batch.pos.clone()
            try:
                baseline = normalize(batch)
            except Exception:
                n_fail += 1; continue
            n = baseline.shape[0]
            emds = []
            for _ in range(NPERTURB):
                pert = orig.numpy() @ R.random().as_matrix().T + np.random.rand(3)
                batch.pos = torch.tensor(pert, dtype=orig.dtype)
                try:
                    emds.append(float(ot.emd2(ot.unif(n), ot.unif(n),
                                              ot.dist(baseline, normalize(batch), 'euclidean'))))
                except Exception:
                    pass
            batch.pos = orig
            if emds:
                results[g].append(np.mean(emds))
            n_done += 1
            if n_done % 100 == 0:
                print(f'  AE {n_done} molecules ({time.time()-t2:.0f}s)', flush=True)
    print(f'AE eval done {time.time()-t2:.0f}s, {n_fail} failures', flush=True)

    summary = {}
    for g in TARGET:
        v = results[g]
        summary[g] = {'mean_emd': float(np.mean(v)) if v else None, 'n': len(v)}
        print('RESULT', g, summary[g], flush=True)
    with open(os.path.join(HARNESS, 'results', 'ae_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('wrote results/ae_results.json', flush=True)


if __name__ == '__main__':
    main()
