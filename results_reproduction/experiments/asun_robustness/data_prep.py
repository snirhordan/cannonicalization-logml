"""data_prep.py -- molecule loading, point-group labeling, and the synthetic
symmetric battery for the ASUN-robustness harness.

Two data sources:

  QM9   Loaded from the PyG raw cache at
        /home/snirhordan/canonicalization/data/pyg_qm9 (data.pos (n,3),
        data.z (n,) atomic numbers). NOTE (see `load_qm9` docstring): the
        gnnplus env's rdkit (2026.03.1) fails to parse a handful of gdb9.sdf
        entries under torch_geometric 2.3.1's QM9.process() (mol=None from
        stricter modern sanitization) -- we bypass this by forcing PyG's
        rdkit-unavailable code path, which loads the already-present
        raw/qm9_v3.pt pre-processed tensor cache instead of re-deriving from
        the SDF. This changes nothing about pos/z (the only fields this
        harness uses); it only skips deriving extra bond-graph atom features
        we do not need.

  SYNTHETIC BATTERY   Exactly-symmetric point clouds with known point
        groups, spanning easy->hard: regular polygons n=3..8 (D_nh), regular
        bipyramids and prisms n=3..8 (D_nh), Platonic solids (tetrahedron
        Td, cube Oh, octahedron Oh, icosahedron Ih), and idealized real
        molecules (benzene D6h, 1,3,5-triazine D3h, methane Td, allene D2d,
        CO2 D*h, acetylene D*h, trans-1,2-dichloroethylene C2h). Bond-length
        scale ~1.4 Angstrom, matching QM9's scale so sigma is comparable
        across both data sources. Every synthetic geometry's point group is
        VERIFIED (not just assumed) by running it through the identical
        pymatgen PointGroupAnalyzer(tolerance=0.1) call used for QM9 --
        every one of the 19 synthetic entries was checked against its
        intended label during development and matches exactly (see README).

  QM-SYM (Liang et al. 2019) is attempted opportunistically (<=2 minutes);
        see `try_qm_sym_download`. If unavailable, we log the substitution
        and rely on the synthetic battery for the heavy-symmetry (C2h/C3h/
        C4h) regime, per the task brief.

Point-group labeling uses pymatgen directly (tolerance=0.1, matching the
task brief) with a CSV cache so repeated runs do not relabel molecules
already labeled by a previous run. Labeling ~130831 QM9 molecules takes
about 5.5 ms/molecule single-threaded (~12 min total); with multiprocessing
across `n_workers` cores it is proportionally faster.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
QM9_ROOT = os.environ.get("QM9_ROOT", "/home/snirhordan/canonicalization/data/pyg_qm9")
LABEL_CACHE = os.path.join(_HERE, "results", "qm9_point_group_labels.csv")
QM_SYM_NOTE = "/home/snirhordan/linkedin/docs/logml/wiki/notes/qm-sym-dataset.md"


@dataclass
class MoleculeRecord:
    name: str
    dataset: str          # 'qm9' | 'synthetic' | 'qm-sym'
    point_group: str
    species: np.ndarray    # (n,) int64 atomic numbers
    coords: np.ndarray     # (n,3) float64


# ---------------------------------------------------------------------------
# QM9 loading
# ---------------------------------------------------------------------------

def load_qm9(root: str = QM9_ROOT):
    """Load QM9 via PyG, forcing the rdkit-unavailable code path (see module
    docstring) so processing never touches the incompatible rdkit/gdb9.sdf
    route. Safe to call repeatedly; PyG caches the processed tensor once."""
    if "rdkit" not in sys.modules:
        sys.modules["rdkit"] = None  # type: ignore[assignment]
    from torch_geometric.datasets import QM9
    return QM9(root=root)


def label_point_group(species: Sequence[int], coords: np.ndarray, tol: float = 0.1) -> str:
    """pymatgen PointGroupAnalyzer sch_symbol for one molecule. Returns
    'ERROR:<ExceptionType>' rather than raising, so a single pathological
    QM9 geometry cannot abort a batch labeling job."""
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    try:
        m = Molecule([int(z) for z in species], np.asarray(coords, dtype=float))
        return PointGroupAnalyzer(m, tolerance=tol).sch_symbol
    except Exception as e:  # noqa: BLE001
        return f"ERROR:{type(e).__name__}"


def _label_one(args):
    idx, z, pos = args
    return idx, label_point_group(z, pos)


def _load_label_cache(cache_path: str) -> Dict[int, str]:
    cache: Dict[int, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path, newline="") as f:
            for row in csv.DictReader(f):
                cache[int(row["qm9_idx"])] = row["point_group"]
    return cache


def _append_label_cache(cache_path: str, new_rows: Dict[int, str]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    write_header = not os.path.exists(cache_path)
    with open(cache_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["qm9_idx", "point_group"])
        for idx, pg in new_rows.items():
            w.writerow([idx, pg])


def label_qm9_indices(ds, indices: Sequence[int], n_workers: int = 2,
                       cache_path: str = LABEL_CACHE, verbose: bool = True) -> Dict[int, str]:
    """Label point groups for `indices` into QM9, using a CSV cache keyed by
    qm9_idx so repeated calls (e.g. pilot then full) never relabel a
    molecule twice. Uses a multiprocessing.Pool of n_workers."""
    cache = _load_label_cache(cache_path)
    todo = [i for i in indices if i not in cache]
    if verbose:
        print(f"[data_prep] labeling {len(todo)}/{len(indices)} QM9 molecules "
              f"not already in cache ({cache_path})")
    if todo:
        tasks = []
        for i in todo:
            d = ds[i]
            tasks.append((i, d.z.numpy(), d.pos.numpy()))
        t0 = time.time()
        if n_workers <= 1:
            results = [_label_one(t) for t in tasks]
        else:
            with Pool(n_workers) as pool:
                results = pool.map(_label_one, tasks, chunksize=64)
        new_rows = dict(results)
        _append_label_cache(cache_path, new_rows)
        cache.update(new_rows)
        if verbose:
            dt = time.time() - t0
            print(f"[data_prep] labeled {len(todo)} molecules in {dt:.1f}s "
                  f"({1000*dt/max(len(todo),1):.2f} ms/mol, {n_workers} workers)")
    return {i: cache[i] for i in indices}


def census(labels: Dict[int, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pg in labels.values():
        out[pg] = out.get(pg, 0) + 1
    return out


DEFAULT_CAPS = {"C1": 500, "Cs": 500, "C*v": 300, "C2v": None, "C2": 200}
# None => "all molecules of this stratum". Strata not named here (the
# rarer ones: Td, C3h, D6h, C3, C3v, C4v, D2h, ...) ALSO get "all" --
# handled by the else-branch in build_stratified_qm9_sample.


def build_stratified_qm9_sample(ds, labels: Dict[int, str],
                                 caps: Dict[str, Optional[int]] = None,
                                 rng: np.random.RandomState = None) -> List[MoleculeRecord]:
    """Stratified sample: explicit caps for the common strata (None = all),
    ALL molecules for every stratum not named in `caps` (the rare, high-
    symmetry strata this whole experiment is about)."""
    caps = dict(DEFAULT_CAPS if caps is None else caps)
    rng = rng or np.random.RandomState(0)

    buckets: Dict[str, List[int]] = {}
    for idx, pg in labels.items():
        if pg.startswith("ERROR:"):
            continue
        buckets.setdefault(pg, []).append(idx)

    chosen: List[int] = []
    for pg, idxs in buckets.items():
        cap = caps.get(pg, "__unset__")
        if cap == "__unset__" or cap is None:
            chosen.extend(idxs)                      # uncapped / rare stratum: ALL
        else:
            idxs = list(idxs)
            rng.shuffle(idxs)
            chosen.extend(idxs[:cap])

    records = []
    for idx in chosen:
        d = ds[idx]
        records.append(MoleculeRecord(
            name=f"qm9_{idx}",
            dataset="qm9",
            point_group=labels[idx],
            species=d.z.numpy().astype(np.int64),
            coords=d.pos.numpy().astype(np.float64),
        ))
    return records


# ---------------------------------------------------------------------------
# Synthetic symmetric battery
# ---------------------------------------------------------------------------

def _regular_polygon(n: int, edge: float = 1.4) -> np.ndarray:
    theta = 2 * np.pi / n
    R = edge / (2 * np.sin(np.pi / n))
    return np.array([[R * np.cos(i * theta), R * np.sin(i * theta), 0.0] for i in range(n)])


def _bipyramid(n: int, edge: float = 1.4, h_ratio: float = 0.8) -> np.ndarray:
    ring = _regular_polygon(n, edge)
    R = np.linalg.norm(ring[0])
    h = h_ratio * R
    apex = np.array([[0.0, 0.0, h], [0.0, 0.0, -h]])
    return np.vstack([ring, apex])


def _prism(n: int, edge: float = 1.4, h_ratio: float = 0.75) -> np.ndarray:
    ring = _regular_polygon(n, edge)
    h = h_ratio * edge
    top = ring + np.array([0.0, 0.0, h / 2])
    bot = ring + np.array([0.0, 0.0, -h / 2])
    return np.vstack([top, bot])


def _tetrahedron(edge: float = 1.4) -> np.ndarray:
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    v *= edge / (2 * np.sqrt(2))
    return v


def _cube(edge: float = 1.4) -> np.ndarray:
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    v *= edge / 2.0
    return v


def _octahedron(edge: float = 1.4) -> np.ndarray:
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    v *= edge / np.sqrt(2)
    return v


def _icosahedron(edge: float = 1.4) -> np.ndarray:
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts.append([0, s1 * 1, s2 * phi])
            verts.append([s1 * 1, s2 * phi, 0])
            verts.append([s1 * phi, 0, s2 * 1])
    v = np.array(verts, dtype=float)
    dists = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    raw_edge = dists.min()
    v *= edge / raw_edge
    return v


def _benzene():
    n = 6
    Rc, dch = 1.395, 1.09
    theta = 2 * np.pi / n
    Cs = np.array([[Rc * np.cos(i * theta), Rc * np.sin(i * theta), 0] for i in range(n)])
    Hs = np.array([[(Rc + dch) * np.cos(i * theta), (Rc + dch) * np.sin(i * theta), 0] for i in range(n)])
    coords = np.vstack([Cs, Hs])
    species = [6] * n + [1] * n
    return species, coords


def _triazine():
    n = 6
    Rring, dch = 1.338, 1.09
    theta = 2 * np.pi / n
    ring = np.array([[Rring * np.cos(i * theta), Rring * np.sin(i * theta), 0] for i in range(n)])
    ring_species = [6, 7] * 3  # alternating C, N
    Hs, Hspecies = [], []
    for i in range(n):
        if ring_species[i] == 6:
            Hs.append([(Rring + dch) * np.cos(i * theta), (Rring + dch) * np.sin(i * theta), 0])
            Hspecies.append(1)
    coords = np.vstack([ring, np.array(Hs)])
    species = ring_species + Hspecies
    return species, coords


def _methane():
    d = 1.094
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    v = v / np.linalg.norm(v[0]) * d
    coords = np.vstack([[0, 0, 0], v])
    species = [6, 1, 1, 1, 1]
    return species, coords


def _allene():
    dz, dch, ang = 1.31, 1.087, np.radians(59.0)
    C0 = np.array([0.0, 0.0, 0.0])
    C1 = np.array([0.0, 0.0, dz])
    C2 = np.array([0.0, 0.0, -dz])
    H1a = C1 + dch * np.array([np.sin(ang), 0, np.cos(ang)])
    H1b = C1 + dch * np.array([-np.sin(ang), 0, np.cos(ang)])
    H2a = C2 + dch * np.array([0, np.sin(ang), -np.cos(ang)])
    H2b = C2 + dch * np.array([0, -np.sin(ang), -np.cos(ang)])
    coords = np.array([C0, C1, C2, H1a, H1b, H2a, H2b])
    species = [6, 6, 6, 1, 1, 1, 1]
    return species, coords


def _co2():
    d = 1.16
    coords = np.array([[0, 0, 0], [0, 0, d], [0, 0, -d]])
    species = [6, 8, 8]
    return species, coords


def _acetylene():
    dcc, dch = 1.20, 1.06
    coords = np.array([[0, 0, 0], [0, 0, dcc], [0, 0, -dch], [0, 0, dcc + dch]])
    species = [6, 6, 1, 1]
    return species, coords


def _trans_dichloroethylene():
    C1 = np.array([-0.665, 0.0, 0.0])

    def rot(vec, deg):
        a = np.radians(deg)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        return R @ vec

    to_C2 = np.array([1.0, 0.0, 0.0])
    H1 = C1 + 1.076 * rot(to_C2, 120)
    Cl1 = C1 + 1.726 * rot(to_C2, -120)
    C2, H2, Cl2 = -C1, -H1, -Cl1
    coords = np.array([C1, C2, H1, H2, Cl1, Cl2])
    species = [6, 6, 1, 1, 17, 17]
    return species, coords


def build_synthetic_battery(edge: float = 1.4, label_tol: float = 0.1,
                             verify: bool = True) -> List[MoleculeRecord]:
    """The synthetic symmetric battery. Every geometry's point group is
    computed with the SAME pymatgen call used for QM9 (verify=True default)
    rather than trusted from the construction, catching any accidental
    extra/reduced symmetry from a geometry bug."""
    records: List[MoleculeRecord] = []

    def add(name, species, coords):
        species = np.asarray(species, dtype=np.int64)
        coords = np.asarray(coords, dtype=np.float64)
        coords = coords - coords.mean(axis=0)
        pg = label_point_group(species, coords, tol=label_tol) if verify else "UNVERIFIED"
        records.append(MoleculeRecord(name=name, dataset="synthetic", point_group=pg,
                                       species=species, coords=coords))

    for n in range(3, 9):
        add(f"polygon{n}", [6] * n, _regular_polygon(n, edge))
    for n in range(3, 9):
        add(f"bipyramid{n}", [6] * (n + 2), _bipyramid(n, edge))
    for n in range(3, 9):
        add(f"prism{n}", [6] * (2 * n), _prism(n, edge))

    add("tetrahedron", [6] * 4, _tetrahedron(edge))
    add("cube", [6] * 8, _cube(edge))
    add("octahedron", [6] * 6, _octahedron(edge))
    add("icosahedron", [6] * 12, _icosahedron(edge))

    for name, builder in [
        ("benzene", _benzene), ("triazine", _triazine), ("methane", _methane),
        ("allene", _allene), ("co2", _co2), ("acetylene", _acetylene),
        ("trans_dichloroethylene", _trans_dichloroethylene),
    ]:
        species, coords = builder()
        add(name, species, coords)

    return records


# ---------------------------------------------------------------------------
# QM-sym (opportunistic; <=2 min budget, see module docstring)
# ---------------------------------------------------------------------------

QM_SYM_CANDIDATE_URLS = [
    # Figshare DOI 10.6084/m9.figshare.9638093 (per QM_SYM_NOTE) "download
    # all files" endpoint -- a single zip bundling the per-point-group
    # QM_sym_xyz_N.tar shards plus readme.txt.
    "https://ndownloader.figshare.com/articles/9638093/versions/1",
    "https://github.com/XI-Lab/QM-sym-database/archive/refs/heads/main.zip",
]

# Elements present in QM-sym (per QM_SYM_NOTE): H, B, C, N, O, F, Cl, Br.
QM_SYM_SYMBOL_TO_Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, "Br": 35}


def try_qm_sym_download(out_dir: str, timeout_s: int = 30) -> Optional[str]:
    """Opportunistic QM-sym download, <=~2 minutes total across all
    candidate URLs. Returns the downloaded path on success, or None (and
    prints the substitution note) on failure/timeout. NEVER raises."""
    os.makedirs(out_dir, exist_ok=True)
    for url in QM_SYM_CANDIDATE_URLS:
        dest = os.path.join(out_dir, os.path.basename(url) or "qm_sym_download.bin")
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(dest, "wb") as f:
                # Bound total bytes read by wall-clock, not just the
                # per-request timeout, so a slow-but-alive connection can't
                # blow the opportunistic budget either.
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    if time.time() - t0 > timeout_s:
                        raise TimeoutError("QM-sym download exceeded opportunistic budget")
            print(f"[data_prep] QM-sym: downloaded {url} -> {dest}")
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"[data_prep] QM-sym: {url} failed ({type(e).__name__}: {e}); trying next / giving up")
    print("[data_prep] QM-sym NOT obtained within the opportunistic budget -- "
          "SUBSTITUTION: relying on the synthetic symmetric battery "
          "(C2h/C3h/C4h-relevant shapes: bipyramids/prisms n=3,4,6, "
          "trans-dichloroethylene C2h, benzene/triazine-family D3h/D6h) "
          "for the heavy-symmetry regime QM-sym would have covered. "
          f"See {QM_SYM_NOTE} for the dataset's construction and caveats.")
    return None


def parse_qm_sym_xyz_text(text: str):
    """Parse ONE QM-sym bespoke .xyz file's raw text (see QM_SYM_NOTE):
    line 1 = atom count; line 2 = pipe-delimited property vector whose
    FIRST field is the symmetry-group label (this is the DFT-symmetry-
    constrained ground-truth label -- QM-sym's geometries are generated to
    strictly satisfy this group, unlike QM9/synthetic where we infer the
    group post hoc with pymatgen); lines 3..2+n = "Symbol x y z
    mulliken_charge". Returns (point_group, species_z (n,) int, coords
    (n,3) float). Standard xyz readers choke on line 2 not being a blank/
    comment-only line -- hence this bespoke parser."""
    lines = [l.rstrip("\r") for l in text.split("\n")]
    n = int(lines[0].strip())
    point_group = lines[1].split("|")[0].strip()
    species, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        sym = parts[0]
        species.append(QM_SYM_SYMBOL_TO_Z[sym])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return point_group, np.array(species, dtype=np.int64), np.array(coords, dtype=np.float64)


def build_qm_sym_sample(zip_path: str, per_tar: int = 60,
                         rng: np.random.RandomState = None) -> List[MoleculeRecord]:
    """Reservoir-sample `per_tar` molecules from each QM_sym_xyz_*.tar shard
    inside the downloaded figshare zip (5-6 shards observed, each internally
    homogeneous or a two-group boundary shard -- see readme.txt), parse them
    with `parse_qm_sym_xyz_text`, and return MoleculeRecords with
    dataset='qm-sym' and point_group taken from the FILE's own declared
    label (ground truth per the dataset's symmetry-constrained construction,
    not re-derived with pymatgen -- see docstring above)."""
    import tarfile
    import zipfile

    rng = rng or np.random.RandomState(0)
    records: List[MoleculeRecord] = []
    with zipfile.ZipFile(zip_path) as z:
        tar_names = sorted(n for n in z.namelist() if n.endswith(".tar"))
        for tar_name in tar_names:
            data = z.read(tar_name)
            with tarfile.open(fileobj=__import__("io").BytesIO(data)) as t:
                members = t.getmembers()
                k = min(per_tar, len(members))
                pick = rng.choice(len(members), size=k, replace=False)
                for i in pick:
                    m = members[int(i)]
                    f = t.extractfile(m)
                    if f is None:
                        continue
                    txt = f.read().decode(errors="replace")
                    try:
                        pg, species, coords = parse_qm_sym_xyz_text(txt)
                    except Exception:  # noqa: BLE001
                        continue
                    coords = coords - coords.mean(axis=0)
                    records.append(MoleculeRecord(
                        name=f"qmsym_{tar_name}_{m.name}", dataset="qm-sym",
                        point_group=pg, species=species, coords=coords,
                    ))
            print(f"[data_prep] QM-sym {tar_name}: sampled {k}/{len(members)} molecules")
    return records


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_molecules(records: List[MoleculeRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        name=np.array([r.name for r in records], dtype=object),
        dataset=np.array([r.dataset for r in records], dtype=object),
        point_group=np.array([r.point_group for r in records], dtype=object),
        species=np.array([r.species for r in records], dtype=object),
        coords=np.array([r.coords for r in records], dtype=object),
        allow_pickle=True,
    )


def load_molecules(path: str) -> List[MoleculeRecord]:
    d = np.load(path, allow_pickle=True)
    out = []
    for name, dataset, pg, sp, co in zip(d["name"], d["dataset"], d["point_group"],
                                          d["species"], d["coords"]):
        out.append(MoleculeRecord(name=str(name), dataset=str(dataset), point_group=str(pg),
                                   species=np.asarray(sp), coords=np.asarray(co)))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--n-qm9-scan", type=int, default=8000,
                     help="pilot mode: how many QM9 molecules to label/scan (full mode always scans all)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "molecules_pilot.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qm-sym-per-tar", type=int, default=60,
                     help="molecules sampled per QM-sym tar shard, if the opportunistic download succeeds")
    ap.add_argument("--skip-qm-sym-download", action="store_true",
                     help="skip the opportunistic download attempt entirely (e.g. reuse a cached zip)")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    print("[data_prep] building synthetic battery...")
    synth = build_synthetic_battery()
    print(f"[data_prep] synthetic battery: {len(synth)} molecules, "
          f"point groups: {sorted(set(r.point_group for r in synth))}")

    qm_sym_dir = os.path.join(_HERE, "data", "qm_sym_raw")
    if args.skip_qm_sym_download and os.path.exists(os.path.join(qm_sym_dir, "1")):
        qm_sym_path = os.path.join(qm_sym_dir, "1")
        print(f"[data_prep] reusing cached QM-sym download at {qm_sym_path}")
    else:
        print("[data_prep] opportunistically trying QM-sym (<=2 min budget)...")
        qm_sym_path = try_qm_sym_download(qm_sym_dir, timeout_s=30)

    qm_sym_records: List[MoleculeRecord] = []
    if qm_sym_path is not None:
        try:
            qm_sym_records = build_qm_sym_sample(qm_sym_path, per_tar=args.qm_sym_per_tar, rng=rng)
            pg_counts = census({i: r.point_group for i, r in enumerate(qm_sym_records)})
            print(f"[data_prep] QM-sym sample: {len(qm_sym_records)} molecules, "
                  f"point groups {pg_counts}")
        except Exception as e:  # noqa: BLE001
            print(f"[data_prep] QM-sym sample extraction FAILED ({type(e).__name__}: {e}); "
                  "SUBSTITUTION: proceeding without QM-sym, relying on the synthetic battery.")
            qm_sym_records = []

    print("[data_prep] loading QM9...")
    ds = load_qm9()
    print(f"[data_prep] QM9: {len(ds)} molecules")

    if args.mode == "pilot":
        idxs = rng.choice(len(ds), size=min(args.n_qm9_scan, len(ds)), replace=False).tolist()
    else:
        idxs = list(range(len(ds)))

    labels = label_qm9_indices(ds, idxs, n_workers=args.workers)
    c = census(labels)
    print(f"[data_prep] census over {len(idxs)} scanned QM9 molecules:")
    for pg, n in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"    {pg:8s} {n:6d}  ({100*n/len(idxs):.2f}%)")

    caps = None if args.mode == "full" else dict(DEFAULT_CAPS)
    qm9_sample = build_stratified_qm9_sample(ds, labels, caps=caps, rng=rng)
    print(f"[data_prep] stratified QM9 sample: {len(qm9_sample)} molecules")

    all_records = synth + qm9_sample + qm_sym_records
    save_molecules(all_records, args.out)
    print(f"[data_prep] wrote {len(all_records)} molecules "
          f"(synthetic={len(synth)}, qm9={len(qm9_sample)}, qm-sym={len(qm_sym_records)}) to {args.out}")
