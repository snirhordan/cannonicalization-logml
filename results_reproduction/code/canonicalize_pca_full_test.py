"""Tests for canonicalize_pca_full: dispatch correctness, never-raises, and
O(3) x S_n (+ translation) invariance across the DISTINCT / AXIAL / TRIPLE
regimes, plus a near-degenerate cloud routed to AXIAL (not DISTINCT).

Run:  /home/snirhordan/miniconda3/envs/gnnplus/bin/python3 canonicalize_pca_full_test.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
from scipy.stats import ortho_group

from canonicalize_pca_full import canonicalize_pca_full, DEFAULT_GAP_RTOL

RNG = np.random.default_rng(20260716)


# ----------------------------------------------------------------- helpers

def sorted_rows(X, decimals=7):
    """Row-lexsort a point cloud (for order-agnostic comparison)."""
    K = np.round(X, decimals)
    idx = np.lexsort(K.T[::-1])
    return X[idx]


def random_rigid_transform(n, rng):
    Q = ortho_group.rvs(dim=3, random_state=rng)
    t = rng.uniform(-3.0, 3.0, size=3)
    perm = rng.permutation(n)
    return Q, t, perm


def apply_rigid(X, Q, t, perm):
    # Column-vector convention x -> Q x  <=>  row convention X @ Q.T
    return (X @ Q.T)[perm] + t


# ----------------------------------------------------- test point clouds

def generic_asymmetric_cloud():
    # A hand-picked asymmetric 5-point cloud: no repeated pairwise distances,
    # no reflective/rotational symmetry -> well-separated covariance
    # spectrum (DISTINCT).
    return np.array([
        [0.0, 0.0, 0.0],
        [1.3, 0.2, 0.1],
        [0.1, 2.1, 0.4],
        [-0.7, 0.3, 1.9],
        [0.9, -1.1, -0.6],
    ], float)


def linear_cloud():
    # Collinear points -> covariance has one nonzero eigenvalue and a
    # degenerate zero-eigenvalue plane orthogonal to it: AXIAL.
    t = np.array([0.3, 0.9, 0.15])
    t /= np.linalg.norm(t)
    s = np.array([-3.0, -1.5, 0.0, 1.2, 2.7, 4.1])
    return s[:, None] * t[None, :]


def benzene_hexagon(r=1.0, z=0.0):
    # Planar regular hexagon (benzene-ring-like): covariance has a doubly
    # degenerate in-plane eigenvalue and a simple (zero, if planar)
    # out-of-plane eigenvalue: AXIAL.
    k = np.arange(6)
    ang = 2 * np.pi * k / 6
    return np.stack([r * np.cos(ang), r * np.sin(ang), np.full(6, z)], axis=1)


def tetrahedron(r=1.0):
    # Regular tetrahedron: fully symmetric covariance (triple-degenerate
    # spectrum): TRIPLE.
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return r * V / np.linalg.norm(V[0])


def near_degenerate_cloud(gap=1e-4):
    # Axial spectrum with a SMALL but nonzero gap between the two nominally
    # degenerate eigenvalues (a slightly squashed hexagon): must still be
    # routed to AXIAL under the default gap_rtol, not DISTINCT.
    base = benzene_hexagon()
    base[:, 1] *= (1.0 + gap)  # break exact in-plane degeneracy slightly
    return base


REGIME_CLOUDS = [
    ("DISTINCT", generic_asymmetric_cloud()),
    ("AXIAL", linear_cloud()),
    ("AXIAL", benzene_hexagon()),
    ("TRIPLE", tetrahedron()),
]


# --------------------------------------------------------------- test runner

class Counter:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append((name, detail))

    def run(self, name, fn):
        try:
            fn(self)
        except Exception:
            self.failed += 1
            self.failures.append((name, traceback.format_exc()))


def test_regime_labels_and_never_raises(c: Counter):
    for expected_regime, X in REGIME_CLOUDS:
        try:
            res = canonicalize_pca_full(X)
            raised = False
        except Exception:
            raised = True
            res = None
        c.check(f"no-raise[{expected_regime}]", not raised)
        if res is not None:
            c.check(
                f"regime-label[{expected_regime}]",
                res.regime == expected_regime,
                f"expected {expected_regime}, got {res.regime}",
            )
            c.check(
                f"shape[{expected_regime}]",
                res.points.shape == X.shape,
                str(res.points.shape),
            )
            c.check(
                f"order-is-permutation[{expected_regime}]",
                sorted(res.order.tolist()) == list(range(X.shape[0])),
            )


def test_invariance(c: Counter):
    for expected_regime, X in REGIME_CLOUDS:
        base = canonicalize_pca_full(X)
        base_sorted = sorted_rows(base.points)
        worst = 0.0
        n_trials = 20
        for _ in range(n_trials):
            Q, t, perm = random_rigid_transform(X.shape[0], RNG)
            Xg = apply_rigid(X, Q, t, perm)
            try:
                out = canonicalize_pca_full(Xg)
            except Exception:
                c.check(f"invariance-no-raise[{expected_regime}]", False,
                        traceback.format_exc())
                continue
            c.check(f"invariance-no-raise[{expected_regime}]", True)
            c.check(
                f"invariance-regime-stable[{expected_regime}]",
                out.regime == expected_regime,
                f"expected {expected_regime}, got {out.regime}",
            )
            out_sorted = sorted_rows(out.points)
            d = np.abs(out_sorted - base_sorted).max()
            worst = max(worst, d)
        c.check(
            f"invariance-agree[{expected_regime}]",
            worst < 1e-7,
            f"worst max-abs-diff over {n_trials} trials = {worst:.3e}",
        )


def test_never_raises_extra(c: Counter):
    # Degenerate / edge clouds that must never raise regardless of regime.
    edge_clouds = [
        np.zeros((4, 3)),                          # R0: all at centroid
        np.array([[0.0, 0.0, 0.0]]),               # single point
        np.eye(3) * 1e-12,                         # tiny-scale generic cloud
        tetrahedron() * 1e6,                       # large-scale TRIPLE
    ]
    for i, X in enumerate(edge_clouds):
        try:
            res = canonicalize_pca_full(X)
            c.check(f"edge-no-raise[{i}]", True)
            c.check(f"edge-shape[{i}]", res.points.shape == X.shape)
        except Exception:
            c.check(f"edge-no-raise[{i}]", False, traceback.format_exc())


def test_near_degenerate_routes_to_axial(c: Counter):
    X = near_degenerate_cloud(gap=1e-4)
    # Sanity: confirm the gap really is within [default routing threshold,
    # far below canonical_pca's own 1e-6 raise threshold] so this is a
    # meaningful test of the LOOSE dispatch threshold, not a fluke.
    Xc = X - X.mean(axis=0)
    lam = np.linalg.eigh(Xc.T @ Xc)[0]
    top = lam[-1]
    rel_gaps = [(lam[1] - lam[0]) / top, (lam[2] - lam[1]) / top]
    c.check(
        "near-degenerate-gap-in-target-band",
        min(rel_gaps) < DEFAULT_GAP_RTOL,
        f"rel_gaps={rel_gaps}, DEFAULT_GAP_RTOL={DEFAULT_GAP_RTOL}",
    )

    try:
        base = canonicalize_pca_full(X)
        raised = False
    except Exception:
        raised = True
        base = None
    c.check("near-degenerate-no-raise", not raised)
    if base is None:
        return
    c.check(
        "near-degenerate-routed-to-axial",
        base.regime == "AXIAL",
        f"got regime={base.regime}",
    )

    # Stability under a small in-plane eigenvector rotation induced by
    # transforming the cloud rigidly: canonicalize_pca_full(X) and
    # canonicalize_pca_full(rigid(X)) must still agree, and both must be
    # AXIAL (this is the noise-robustness property under test).
    base_sorted = sorted_rows(base.points)
    worst = 0.0
    for _ in range(20):
        Q, t, perm = random_rigid_transform(X.shape[0], RNG)
        Xg = apply_rigid(X, Q, t, perm)
        out = canonicalize_pca_full(Xg)
        c.check("near-degenerate-invariance-regime-stable",
               out.regime == "AXIAL", f"got {out.regime}")
        out_sorted = sorted_rows(out.points)
        worst = max(worst, np.abs(out_sorted - base_sorted).max())
    c.check(
        "near-degenerate-invariance-agree",
        worst < 1e-7,
        f"worst max-abs-diff = {worst:.3e}",
    )


def main():
    c = Counter()
    c.run("regime-labels-and-never-raises", test_regime_labels_and_never_raises)
    c.run("invariance", test_invariance)
    c.run("never-raises-extra", test_never_raises_extra)
    c.run("near-degenerate-routes-to-axial", test_near_degenerate_routes_to_axial)

    print(f"PASS: {c.passed}  FAIL: {c.failed}")
    if c.failures:
        print("\nFailures:")
        for name, detail in c.failures:
            print(f"  - {name}")
            if detail:
                for line in str(detail).splitlines():
                    print(f"      {line}")
    return 0 if c.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
