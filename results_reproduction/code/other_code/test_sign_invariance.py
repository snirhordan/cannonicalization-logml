"""Tests for the revised sign_invariance.py.

Run:  python3 test_sign_invariance.py   (pytest-compatible; no pytest needed)

Contains the original student tests (adapted: canonical_pca now RAISES on a
degenerate spectrum instead of silently misbehaving, canonical_polar centers
by default, and the reflection test is named for what it asserts) plus one
regression test per day-2 review finding (wiki/notes/polar-2d-canonicalization.md).
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
from scipy.stats import ortho_group

from sign_invariance import canonical_pca, canonical_polar

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------- helpers

def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def set_dist(A, B):
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return max(D.min(axis=1).max(), D.min(axis=0).max())


def is_nondegenerate(eigenvalues, rtol=1e-6):
    lam = np.sort(eigenvalues)
    return bool(np.all(np.diff(lam) > rtol * max(lam[-1], 1e-300)))


def orbit_copies(X, rng, n_copies, dim, translate=True):
    for _ in range(n_copies):
        Q = ortho_group.rvs(dim=dim, random_state=rng)
        t = rng.uniform(-3, 3, size=dim) if translate else np.zeros(dim)
        yield X[rng.permutation(len(X))] @ Q + t


# ------------------------------------------------- original tests (adapted)

def test_pca_invariance_generic():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(200):
        n = int(rng.integers(4, 20))
        X = rng.normal(size=(n, 3))
        Xc = X - X.mean(axis=0)
        if not is_nondegenerate(np.linalg.eigvalsh(Xc.T @ Xc)):
            continue                      # degeneracy checked BEFORE the call
        checked += 1
        result = canonical_pca(X)
        for Xg in orbit_copies(X, rng, 3, dim=3):
            np.testing.assert_allclose(result, canonical_pca(Xg), atol=1e-8)
    assert checked > 0
    print(f"    ({checked}/200 non-degenerate trials checked)")


def test_polar_invariance_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 20))
        X = rng.normal(size=(n, 2))
        result = canonical_polar(X)
        Xc = X - X.mean(axis=0)
        np.testing.assert_allclose(                    # geometry preserved
            np.sort(np.linalg.norm(Xc, axis=1)),
            np.sort(np.linalg.norm(result, axis=1)), atol=1e-8)
        d0 = np.sort(np.linalg.norm(Xc[:, None] - Xc[None, :], axis=-1).ravel())
        d1 = np.sort(np.linalg.norm(result[:, None] - result[None, :], axis=-1).ravel())
        np.testing.assert_allclose(d0, d1, atol=1e-8)
        # canonical start on the +y ray (first non-origin row)
        r = np.linalg.norm(result, axis=1)
        lead = result[np.flatnonzero(r > 1e-9 * max(r.max(), 1e-300))[0]]
        assert abs(lead[0]) <= 2e-9 * max(r.max(), 1.0) and lead[1] > 0
        for Xg in orbit_copies(X, rng, 3, dim=2):
            np.testing.assert_allclose(result, canonical_polar(Xg), atol=1e-8)


def test_polar_invariance_on_symmetric_clouds():
    rng = np.random.default_rng(1)
    for k in (3, 4, 6):
        angles = TWO_PI * np.arange(k) / k
        polygon = np.column_stack([np.cos(angles), np.sin(angles)])
        nested = np.vstack([polygon, 2.5 * polygon @ rotation(np.pi / k)])
        for X in (polygon, nested):
            result = canonical_polar(X)
            for Xg in orbit_copies(X, rng, 20, dim=2, translate=True):
                np.testing.assert_allclose(result, canonical_polar(Xg),
                                           atol=1e-8)


def test_polar_reflection_maps_to_same_canonical_form():
    # (renamed from test_polar_distinguishes_reflections, which asserted the
    # opposite of its name)
    X = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, -1.0], [0.5, 0.5]])
    np.testing.assert_allclose(canonical_polar(X),
                               canonical_polar(X * np.array([1.0, -1.0])),
                               atol=1e-9)


# --------------------------------------------------- regressions: PCA strata

def test_pca_degenerate_spectrum_raises():
    # equilateral triangle in the xy-plane: spectrum {0, l, l}
    th = TWO_PI / 3
    X = np.stack([rotation(th * k) @ np.array([1.3, 0.4]) for k in range(3)])
    X3 = np.column_stack([X, np.zeros(3)])
    try:
        canonical_pca(X3)
        raise AssertionError("expected ValueError on degenerate spectrum")
    except ValueError:
        pass


def _tied_column_cloud(rng):
    y1 = np.array([3., -3., 1., -1., 2., -2., .5, -.5])   # sorted(v)==sorted(-v)
    a, b = rng.normal(size=8), rng.normal(size=8)
    y2 = a - (a @ y1) / (y1 @ y1) * y1
    y3 = b - (b @ y1) / (y1 @ y1) * y1
    y3 = y3 - (y3 @ y2) / (y2 @ y2) * y2
    y1, y2, y3 = (5 * y1 / np.linalg.norm(y1), 2 * y2 / np.linalg.norm(y2),
                  8 * y3 / np.linalg.norm(y3))
    return np.column_stack([y1, y2, y3])


def test_pca_sign_tie_cloud_invariant():
    # stratum 2: negation-symmetric column, distinct eigenvalues, asymmetric
    # cloud -- the old per-column rule failed 51% of poses; the joint argmin
    # must be exactly invariant
    rng = np.random.default_rng(3)
    X = _tied_column_cloud(rng)
    base = canonical_pca(X)
    for Xg in orbit_copies(X, rng, 100, dim=3):
        np.testing.assert_allclose(base, canonical_pca(Xg), atol=1e-8)


def test_pca_mirror_and_c2_symmetric_clouds_invariant():
    # self-symmetric clouds: argmin candidates tie with EQUAL matrices
    rng = np.random.default_rng(4)
    aa = np.array([3., 1., 2., .5])
    B = rng.normal(size=4); C = rng.normal(size=4)
    C = C - (C @ B) / (B @ B) * B
    mirror = np.column_stack([np.concatenate([aa, -aa]),
                              np.concatenate([B, B]),
                              np.concatenate([C, C])])
    bb = rng.normal(size=4); bb = bb - (bb @ aa) / (aa @ aa) * aa
    cc = rng.normal(size=4)
    c2 = np.column_stack([np.concatenate([aa, -aa]),
                          np.concatenate([bb, -bb]),
                          np.concatenate([cc, cc])])
    for X in (mirror, c2):
        base = canonical_pca(X, assume_centered=True)
        for _ in range(100):
            Q = ortho_group.rvs(dim=3, random_state=rng)
            P = rng.permutation(len(X))
            np.testing.assert_allclose(
                base, canonical_pca(X[P] @ Q, assume_centered=True), atol=1e-8)


def test_pca_duplicate_sort_keys_invariant():
    # stratum 4: two points sharing their leading principal coordinate --
    # raw-float lexsort ordered them by eigh noise (213/500 failures before)
    rng = np.random.default_rng(7)
    c0 = np.array([1.7, 1.7, -0.4, 2.6, -3.1, -2.5])
    u = rng.normal(size=6); u -= (u @ c0) / (c0 @ c0) * c0
    w = rng.normal(size=6); w -= (w @ c0) / (c0 @ c0) * c0
    w -= (w @ u) / (u @ u) * u
    X = np.column_stack([c0 / np.linalg.norm(c0), 3 * u / np.linalg.norm(u),
                         7 * w / np.linalg.norm(w)])
    base = canonical_pca(X, assume_centered=True)
    for _ in range(200):
        Q = ortho_group.rvs(dim=3, random_state=rng)
        P = rng.permutation(len(X))
        np.testing.assert_allclose(base,
                                   canonical_pca(X[P] @ Q, assume_centered=True),
                                   atol=1e-8)


def test_pca_translation_and_shapes():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(9, 3))
    if is_nondegenerate(np.linalg.eigvalsh((X - X.mean(0)).T @ (X - X.mean(0)))):
        np.testing.assert_allclose(canonical_pca(X),
                                   canonical_pca(X + np.array([5., -3., 2.])),
                                   atol=1e-8)
    # d = 2 and d = 4 now work (previously: crash / silent non-invariance)
    for d in (2, 4):
        Xd = rng.normal(size=(10, d))
        Xdc = Xd - Xd.mean(axis=0)
        if not is_nondegenerate(np.linalg.eigvalsh(Xdc.T @ Xdc)):
            continue
        base = canonical_pca(Xd)
        for Xg in orbit_copies(Xd, rng, 20, dim=d):
            np.testing.assert_allclose(base, canonical_pca(Xg), atol=1e-8)
    # n = 0 raises cleanly
    try:
        canonical_pca(np.zeros((0, 3)))
        raise AssertionError("expected ValueError on empty input")
    except ValueError:
        pass
    # all-points-identical collapses to zeros
    np.testing.assert_allclose(canonical_pca(np.full((4, 3), 2.5)), 0.0)


# ------------------------------------------------ regressions: polar F1-F7

def test_polar_tiny_and_huge_scale():
    # F1/F3: whole-cloud collapse at radius <= 1e-9 and float-noise trouble
    # at 1e8 are both gone with relative quantization
    rng = np.random.default_rng(6)
    angles = TWO_PI * np.arange(6) / 6
    hexagon = np.column_stack([np.cos(angles), np.sin(angles)])
    scalene = np.array([[1.0, 0.2], [-0.4, 1.1], [-0.9, -0.7], [0.3, -0.8]])
    for shape in (hexagon, scalene):
        for scl in (1e-10, 1e-9, 1.0, 1e8):
            X = shape * scl
            base = canonical_polar(X, assume_centered=True)
            for _ in range(30):
                Q = ortho_group.rvs(dim=2, random_state=rng)
                d = set_dist(base, canonical_polar(X @ Q, assume_centered=True))
                assert d < 1e-6 * scl, (scl, d)


def test_polar_origin_block_permutation():
    # F4: distinct sub-quantum origin points may keep input order (documented,
    # sub-quantum); EXACT-zero duplicates and generic permutations must be
    # exactly invariant
    rng = np.random.default_rng(8)
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.3], [-0.7, 1.4], [0.2, -1.1]])
    Xc = X - X.mean(axis=0)   # exact zeros move off-centroid: use raw + centering
    base = canonical_polar(X)
    for _ in range(50):
        perm = rng.permutation(len(X))
        Q = ortho_group.rvs(dim=2, random_state=rng)
        np.testing.assert_allclose(base, canonical_polar(X[perm] @ Q), atol=1e-9)
    del Xc


def test_polar_same_angle_stacks():
    # radial stacks: several points sharing an angle (zero gaps), plus a
    # duplicated point -- quantized sort keys keep the order pose-stable
    rng = np.random.default_rng(9)
    X = np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0],
                  [0.0, 1.5], [0.0, 1.5], [-1.2, 0.7]])
    base = canonical_polar(X, assume_centered=True)
    for _ in range(50):
        perm = rng.permutation(len(X))
        Q = ortho_group.rvs(dim=2, random_state=rng)
        d = set_dist(base, canonical_polar(X[perm] @ Q, assume_centered=True))
        assert d < 1e-8, d


def test_polar_near_symmetric_jitter_bounded():
    # F2: near-ties from near-symmetry must produce bounded jitter, not flips
    rng = np.random.default_rng(10)
    for eps in (1e-12, 1e-7):
        angles = TWO_PI * np.arange(4) / 4 + np.array([0.0, eps, 0.0, 0.0])
        X = np.column_stack([np.cos(angles), np.sin(angles)])
        base = canonical_polar(X, assume_centered=True)
        worst = 0.0
        for _ in range(50):
            Q = ortho_group.rvs(dim=2, random_state=rng)
            worst = max(worst, set_dist(base,
                        canonical_polar(X @ Q, assume_centered=True)))
        assert worst < max(10 * eps, 1e-8), (eps, worst)


def test_polar_orientation_boundary_regression():
    """Day-3 attack Break B: hexagon at exact 60-degree angles with one
    radius outlier. With absolute-angle quantization the two ORIENTATION
    strings differed by one pose-dependent quantum and a rotation near pi
    flipped the mirror choice macroscopically (deviation 0.5). Gap-based
    quantization must keep every pose consistent."""
    adeg = np.arange(6) * 60.0
    rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    X = np.column_stack([rad * np.cos(np.radians(adeg)),
                         rad * np.sin(np.radians(adeg))])
    base = canonical_polar(X, assume_centered=True)
    # the exact adversarial rotation found by the attack agent, plus a sweep
    for theta in [3.1405548111074495] + list(np.linspace(0, TWO_PI, 60)):
        out = canonical_polar(X @ rotation(theta), assume_centered=True)
        assert np.abs(out - base).max() < 1e-8, theta
    rng = np.random.default_rng(11)
    for _ in range(100):
        Q = ortho_group.rvs(dim=2, random_state=rng)
        out = canonical_polar(X[rng.permutation(6)] @ Q, assume_centered=True)
        assert np.abs(out - base).max() < 1e-8


def test_polar_translation_precision_floor_documented():
    """Day-3 attack Break A: an O(1) translation of a 1e-10-scale cloud
    destroys the cloud's digits in the input representation itself (float64).
    Invariance holds to ~eps*|t| ABSOLUTE (here ~1e-15), which is large
    RELATIVE to the tiny cloud -- a documented precision floor, not a logic
    bug. Assert the absolute bound, and full relative precision when the
    translation is at cloud scale."""
    poly = ring_scaled = np.column_stack(
        [np.cos(TWO_PI * np.arange(3) / 3),
         np.sin(TWO_PI * np.arange(3) / 3)]) * 1e-10
    base = canonical_polar(poly)
    Q = np.array([[0.8, -0.6], [0.6, 0.8]])
    out_big_t = canonical_polar(poly @ Q + np.array([1.7, -2.3]))
    assert np.abs(out_big_t - base).max() < 5e-15          # absolute floor
    out_small_t = canonical_polar(poly @ Q + np.array([1.7e-10, -2.3e-10]))
    assert np.abs(out_small_t - base).max() < 1e-16 + 1e-15 * 1e-10
    del ring_scaled


def test_polar_degenerate_and_edge_inputs():
    # all points at the centroid -> zeros (invariant), n=1 -> zeros
    np.testing.assert_allclose(canonical_polar(np.full((3, 2), 4.0)), 0.0)
    np.testing.assert_allclose(canonical_polar(np.array([[3.0, 4.0]])), 0.0)
    # n=1 about a known origin: lands on +y
    out = canonical_polar(np.array([[2.0, 0.0]]), assume_centered=True)
    np.testing.assert_allclose(out, [[0.0, 2.0]], atol=1e-12)
    for bad in (np.zeros((0, 2)), np.zeros(3), np.zeros((3, 3))):
        try:
            canonical_polar(bad)
            raise AssertionError(f"expected ValueError for shape {bad.shape}")
        except ValueError:
            pass


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
