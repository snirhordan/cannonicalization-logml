"""Tests for canonicalize_3d: invariance under O(3) x S_n (+ translations)
on symmetric configurations, multi-radius (nested-shell) clouds, clouds with
points at the center, random clouds, and noisy versions of all of these.

Run:  python3 test_canonicalize_3d.py   (pytest-compatible; no pytest needed)
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
from scipy.stats import ortho_group

from canonicalize_3d import canonicalize_3d, canonicalize_3d_wwv

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------- helpers

def set_dist(A, B):
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return max(D.min(axis=1).max(), D.min(axis=0).max())


def pairwise(X):
    D = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    return np.sort(D[np.triu_indices(len(X), k=1)])


def check_invariance(X, rng, trials=40, tol_rel=1e-6, translate=True,
                     fn=canonicalize_3d):
    """fn(P X Q + t) must equal fn(X) as a matrix."""
    base = fn(X)
    scale = max(np.linalg.norm(X - X.mean(0), axis=1).max(), 1e-300)
    worst = 0.0
    for _ in range(trials):
        Q = ortho_group.rvs(dim=3, random_state=rng)
        t = rng.uniform(-3, 3, size=3) if translate else np.zeros(3)
        Xg = X[rng.permutation(len(X))] @ Q + t
        out = fn(Xg)
        worst = max(worst, np.abs(out.points - base.points).max())
    assert worst < tol_rel * scale, (worst, scale)
    return base


def check_representative(X, res, tol=1e-8):
    Xc = X - X.mean(axis=0)
    np.testing.assert_allclose(pairwise(Xc), pairwise(res.points), atol=tol)
    assert np.abs(res.points.mean(axis=0)).max() < tol
    # order is the permutation placing input rows into canonical positions
    assert sorted(res.order.tolist()) == list(range(len(X)))


# ----------------------------------------------------- symmetric configs

def tetrahedron(r=1.0):
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return r * V / np.sqrt(3)


def octahedron(r=1.0):
    return r * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], float)


def cube(r=1.0):
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                  for sz in (-1, 1)], float)
    return r * V / np.sqrt(3)


def ring3d(k, r=1.0, z=0.0, phase=0.0):
    a = TWO_PI * np.arange(k) / k + phase
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.full(k, z)])


def c4v_pyramid():
    return np.vstack([ring3d(4, r=1.0, z=0.0), [[0.0, 0.0, 1.3]]])


def c3v_config():
    return np.vstack([ring3d(3, r=1.2, z=-0.2), [[0.0, 0.0, 1.5]]])


def test_symmetric_configs_invariant():
    rng = np.random.default_rng(0)
    for X, expected_regime in (
        (tetrahedron(), "R3"), (octahedron(), "R3"), (cube(), "R3"),
        (ring3d(6), "R2"),                 # planar hexagon: axis = normal
        (c4v_pyramid(), "R2"), (c3v_config(), "R2"),
    ):
        res = check_invariance(X, rng)
        check_representative(X, res)
        assert res.regime == expected_regime, (expected_regime, res.regime)


def test_many_radii_nested_shells():
    rng = np.random.default_rng(1)
    # nested octahedra at several radii (kept symmetric: R3)
    X = np.vstack([octahedron(r) for r in (0.5, 1.0, 2.0, 5.0)])
    res = check_invariance(X, rng)
    check_representative(X, res)
    assert res.regime == "R3"
    # nested rings at different radii and heights, C6 symmetric (R2)
    Y = np.vstack([ring3d(6, r=1.0, z=0.3), ring3d(6, r=2.5, z=-0.3,
                                                   phase=np.pi / 6)])
    res = check_invariance(Y, rng)
    check_representative(Y, res)
    assert res.regime == "R2"
    # nested shells with mutually incompatible symmetry (tetra + rotated octa)
    Z = np.vstack([tetrahedron(1.0), octahedron(2.0) @ ortho_group.rvs(
        dim=3, random_state=rng)])
    res = check_invariance(Z, rng)
    check_representative(Z, res)


def test_points_at_center():
    rng = np.random.default_rng(2)
    for base_cfg in (tetrahedron(), ring3d(5), rng.normal(size=(7, 3))):
        X = np.vstack([base_cfg - base_cfg.mean(0), np.zeros((2, 3))])
        X = X - X.mean(axis=0)   # recenter so the added points sit AT centroid
        res = check_invariance(X, rng)
        check_representative(X, res)


def test_random_clouds():
    rng = np.random.default_rng(3)
    for n in (2, 3, 5, 9, 20, 40):
        X = rng.normal(size=(n, 3))
        res = check_invariance(X, rng, trials=25)
        check_representative(X, res)
    # generic clouds should take the non-degenerate route
    assert canonicalize_3d(rng.normal(size=(15, 3))).regime == "R1"


def test_noisy_configs():
    # a noisy cloud is just a fixed cloud: canonicalization must still be
    # exactly invariant across poses of THAT cloud, at every noise level
    rng = np.random.default_rng(4)
    for X0 in (tetrahedron(), ring3d(6), c4v_pyramid(),
               np.vstack([octahedron(1.0), octahedron(3.0)])):
        for eps in (1e-8, 1e-4, 1e-2):
            X = X0 + eps * rng.normal(size=X0.shape)
            res = check_invariance(X, rng, trials=25)
            check_representative(X, res)


def test_scales():
    rng = np.random.default_rng(5)
    for scl in (1e-8, 1.0, 1e6):
        for X0 in (tetrahedron(), c4v_pyramid()):
            X = X0 * scl
            base = canonicalize_3d(X)
            for _ in range(20):
                Q = ortho_group.rvs(dim=3, random_state=rng)
                out = canonicalize_3d(X[rng.permutation(len(X))] @ Q)
                assert np.abs(out.points - base.points).max() < 1e-6 * scl


def test_reflection_of_chiral_cloud():
    # a chiral cloud and its mirror image must canonicalize identically
    rng = np.random.default_rng(6)
    X = rng.normal(size=(8, 3))
    M = X * np.array([1.0, 1.0, -1.0])
    np.testing.assert_allclose(canonicalize_3d(X).points,
                               canonicalize_3d(M).points, atol=1e-8)


def test_edge_cases():
    # single point / all identical -> zeros
    np.testing.assert_allclose(canonicalize_3d(np.array([[1., 2., 3.]])).points, 0.0)
    res = canonicalize_3d(np.full((4, 3), 7.0))
    np.testing.assert_allclose(res.points, 0.0)
    assert res.regime == "R0"
    # collinear cloud (R2 with zero pair); asymmetric spacing
    rng = np.random.default_rng(7)
    t = np.array([-2.0, -0.5, 1.0, 1.5])[:, None]
    X = t * np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2)
    res = check_invariance(X, rng, trials=25)
    assert res.regime == "R2"
    # coplanar random cloud (rank 2, generically distinct nonzero eigenvalues)
    Xp = np.column_stack([rng.normal(size=(9, 2)), np.zeros(9)])
    check_invariance(Xp, rng, trials=25)
    # malformed inputs
    for bad in (np.zeros((0, 3)), np.zeros(3), np.zeros((3, 2))):
        try:
            canonicalize_3d(bad)
            raise AssertionError(f"expected ValueError for {bad.shape}")
        except ValueError:
            pass


def test_near_collinear_dust_bounded():
    """Day-3 review FATAL regression: collinear cloud with one perpendicular
    kick in the (quantum, axial_rtol) window. The frame completion is not
    equivariant, so the canonical CHOICE must come from projected keys
    (pose-stable) and outputs may differ only by ~the dust size -- the
    original code differed by 2x the cloud scale under a 90-degree rotation."""
    rng = np.random.default_rng(9)
    kick = 1e-8
    X = np.array([[0., 0., -1.2], [0., 0., -0.4], [0., 0., 0.5],
                  [0., 0., 1.0], [kick, 0., 1.2]])
    base = canonicalize_3d(X)
    scale = np.linalg.norm(X - X.mean(0), axis=1).max()
    # the reviewer's exact adversarial pose: pure 90-degree rotation about y
    Ry = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    out = canonicalize_3d(X @ Ry)
    assert np.abs(out.points - base.points).max() < 10 * kick * scale + 1e-12
    for _ in range(40):
        Q = ortho_group.rvs(dim=3, random_state=rng)
        out = canonicalize_3d(X[rng.permutation(len(X))] @ Q)
        assert np.abs(out.points - base.points).max() < 10 * kick * scale + 1e-12


def test_tiny_absolute_scale_not_r0():
    """Day-3 review MAJOR regression: a tetrahedron at radius 1e-15 about the
    origin is well-resolved relative to its own scale and must canonicalize
    as R3, not collapse to the all-at-centroid branch."""
    rng = np.random.default_rng(10)
    for r in (1e-15, 1e-16):
        X = tetrahedron(r)
        res = canonicalize_3d(X)
        assert res.regime == "R3", (r, res.regime)
        base = res.points
        for _ in range(20):
            Q = ortho_group.rvs(dim=3, random_state=rng)
            out = canonicalize_3d(X[rng.permutation(4)] @ Q)
            assert np.abs(out.points - base).max() < 1e-6 * r


def test_wwv_variant_invariance():
    """The PCA-free proposal (canonicalize_3d_wwv) must be invariant on the
    full config matrix: symmetric solids, rings, nested shells, random,
    noisy, scaled, chiral pairs, and edge cases -- with regime 'WWV'."""
    rng = np.random.default_rng(12)
    configs = [tetrahedron(), octahedron(), cube(), ring3d(6), c4v_pyramid(),
               c3v_config(), np.vstack([octahedron(r) for r in (0.5, 1., 2.)]),
               rng.normal(size=(2, 3)), rng.normal(size=(7, 3)),
               rng.normal(size=(20, 3)),
               tetrahedron() + 1e-4 * rng.normal(size=(4, 3)),
               ring3d(6) + 1e-8 * rng.normal(size=(6, 3))]
    for X in configs:
        res = check_invariance(X, rng, trials=25, fn=canonicalize_3d_wwv)
        check_representative(X, res)
        assert res.regime == "WWV"
    # scales
    for scl in (1e-8, 1e6):
        X = c4v_pyramid() * scl
        base = canonicalize_3d_wwv(X)
        for _ in range(15):
            Q = ortho_group.rvs(dim=3, random_state=rng)
            out = canonicalize_3d_wwv(X[rng.permutation(len(X))] @ Q)
            assert np.abs(out.points - base.points).max() < 1e-6 * scl
    # chirality: mirror image canonicalizes identically
    X = rng.normal(size=(8, 3))
    np.testing.assert_allclose(canonicalize_3d_wwv(X).points,
                               canonicalize_3d_wwv(X * [1., 1., -1.]).points,
                               atol=1e-8)
    # edges: all-identical -> zeros; collinear cloud invariant
    np.testing.assert_allclose(canonicalize_3d_wwv(np.full((4, 3), 7.0)).points, 0.0)
    t = np.array([-2.0, -0.5, 1.0, 1.5])[:, None]
    Xc = t * np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2)
    check_invariance(Xc, rng, trials=20, fn=canonicalize_3d_wwv)
    # determinism
    X = rng.normal(size=(10, 3))
    a, b = canonicalize_3d_wwv(X), canonicalize_3d_wwv(X)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.order, b.order)


def test_determinism():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(10, 3))
    a, b = canonicalize_3d(X), canonicalize_3d(X)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.order, b.order)


# ------------------------------------------------------------------ runner

def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
