"""Tests for canonicalize_2d: symmetric rings, random rings, adversarial cases.

Run with:  python3 test_canonicalize_2d.py   (no pytest needed; pytest also works)

Test plan
---------
1.  Booth's algorithm against brute-force lex-min over all rotations.
2.  Regular k-gons (k=3..12): SE(2) invariance, detected rotational order = k.
3.  C_m-symmetric non-regular motif rings (m=2,3,4,6): SE(2) invariance,
    detected order = m.
4.  Random concentric clouds with centroid exactly at the ring center
    (constructed so sum of unit vectors = 0): SE(2) invariance,
    representativeness, canonical-ordering invariance.
5.  Random rings about the origin (centroid != center): SO(2) invariance
    with assume_centered=True.
6.  ADVERSARIAL: the (30,100,30,200)-gap cloud where the naive maximal-index
    rule provably picks different physical points under rotation -> naive
    fails, fixed algorithm succeeds.
7.  Naive "random choice" case (regular polygon): set-level benign,
    but fixed algorithm is deterministic (a function).
8.  ADVERSARIAL: scalene triangle on a circle -- centroid != circumcenter,
    so the ring assumption fails after centering (must raise).
9.  Edge cases: n=1, n=2 (antipodal), coincident points, all-points-identical,
    malformed input.
10. Near-symmetric stability: perturbed square at sub-quantum and
    super-quantum perturbation scales.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from canonicalize_2d import (
    canonicalize_2d,
    canonicalize_2d_naive,
    least_rotation_index,
)

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------- helpers

def rot(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def set_dist(A: np.ndarray, B: np.ndarray) -> float:
    """Bidirectional Hausdorff distance between two point sets."""
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return max(D.min(axis=1).max(), D.min(axis=0).max())


def pairwise_dists(X: np.ndarray) -> np.ndarray:
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(len(X), k=1)
    return np.sort(D[iu])


def ring_from_cw_angles(theta_cw_deg, radius=1.0) -> np.ndarray:
    """Points on a circle about the ORIGIN from angles measured clockwise
    from the +y axis, in degrees (the convention of the algorithm)."""
    phi = np.deg2rad(90.0 - np.asarray(theta_cw_deg, dtype=float))
    return radius * np.stack([np.cos(phi), np.sin(phi)], axis=1)


def random_se2_copies(P, rng, n_copies, translate=True):
    """Random rotated (+ translated) copies of P, with rows also shuffled
    to make sure nothing depends on input row order."""
    out = []
    for _ in range(n_copies):
        R = rot(rng.uniform(0.0, TWO_PI))
        t = rng.uniform(-5.0, 5.0, size=2) if translate else np.zeros(2)
        Q = P @ R.T + t
        perm = rng.permutation(len(P))
        out.append((Q[perm], perm))
    return out


def sum_zero_ring(k, rng, radius=1.0, min_gap_deg=1.0):
    """Random k points on a circle of given radius about the origin whose
    CENTROID IS EXACTLY THE CENTER: choose k-2 angles at random, then solve
    the last two unit vectors so that sum(e^{i phi}) = 0. Rejection-sample
    until solvable (|partial sum| <= 2) and all gaps exceed min_gap_deg."""
    assert k >= 3
    for _ in range(10000):
        phi = rng.uniform(0.0, TWO_PI, size=k - 2)
        S = np.array([np.cos(phi).sum(), np.sin(phi).sum()])
        norm_S = np.linalg.norm(S)
        if norm_S > 2.0 - 1e-6:
            continue
        base = np.arctan2(-S[1], -S[0])
        delta = np.arccos(norm_S / 2.0)
        angles = np.concatenate([phi, [base - delta, base + delta]])
        gaps = np.diff(np.sort(np.mod(angles, TWO_PI)))
        wrap = TWO_PI - np.sort(np.mod(angles, TWO_PI))[-1] + np.sort(
            np.mod(angles, TWO_PI))[0]
        if min(gaps.min(initial=wrap), wrap) < np.deg2rad(min_gap_deg):
            continue
        X = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        assert np.linalg.norm(X.mean(axis=0)) < 1e-12
        return X
    raise RuntimeError("rejection sampling failed")


# ------------------------------------------------------------------- tests

def test_booth_vs_bruteforce():
    rng = np.random.default_rng(7)
    for _ in range(300):
        n = int(rng.integers(1, 40))
        s = list(rng.integers(0, 4, size=n))  # small alphabet -> many ties
        rotations = [tuple(s[i:] + s[:i]) for i in range(n)]
        best = min(rotations)
        b = least_rotation_index(s)
        assert rotations[b] == best, (s, b)
        # Booth should return the smallest index achieving the minimum
        # (determinism of the canonical ordering relies only on SOME
        # deterministic choice; smallest-index is what this impl gives).
        assert b == rotations.index(best), (s, b)


def test_regular_polygons():
    rng = np.random.default_rng(0)
    for k in range(3, 13):
        P = ring_from_cw_angles(np.arange(k) * 360.0 / k, radius=2.5)
        res0 = canonicalize_2d(P)
        assert res0.rotational_order == k, (k, res0.rotational_order)
        # canonical start point sits on the +y ray
        p0 = res0.points[res0.order[0]]
        assert abs(p0[0]) < 1e-9 and p0[1] > 0
        for Q, _ in random_se2_copies(P, rng, 50):
            res = canonicalize_2d(Q)
            assert set_dist(res0.points, res.points) < 1e-9, k


def test_symmetric_motifs():
    # C_m-symmetric but NOT regular: motif at (0, 25) degrees repeated m times
    rng = np.random.default_rng(1)
    for m in (2, 3, 4, 6):
        step = 360.0 / m
        angles = np.concatenate([[j * step, j * step + 25.0] for j in range(m)])
        P = ring_from_cw_angles(angles, radius=1.3)  # centroid = 0 by symmetry
        res0 = canonicalize_2d(P)
        assert res0.rotational_order == m, (m, res0.rotational_order)
        for Q, _ in random_se2_copies(P, rng, 50):
            res = canonicalize_2d(Q)
            assert set_dist(res0.points, res.points) < 1e-9, m


def test_random_sum_zero_clouds():
    rng = np.random.default_rng(2)
    for k in (3, 4, 5, 8, 13, 21, 34):
        P = sum_zero_ring(k, rng, radius=1.7)
        res0 = canonicalize_2d(P)
        # representativeness: canonical pose is a rigid motion of the input
        assert np.allclose(pairwise_dists(P), pairwise_dists(res0.points),
                           atol=1e-9)
        assert np.linalg.norm(res0.points.mean(axis=0)) < 1e-9
        p0 = res0.points[res0.order[0]]
        assert abs(p0[0]) < 1e-9 and p0[1] > 0
        for Q, perm in random_se2_copies(P, rng, 20):
            res = canonicalize_2d(Q)
            assert set_dist(res0.points, res.points) < 1e-9, k
            if res0.rotational_order == 1:
                # asymmetric cloud: canonical ORDERING names the same
                # physical points regardless of pose / row order
                assert np.array_equal(perm[res.order], res0.order), k


def test_random_rings_assume_centered():
    # generic rings about the origin: centroid != center, so use SO(2) mode
    rng = np.random.default_rng(3)
    for k in (2, 3, 6, 11, 25, 40):
        phi = np.sort(rng.uniform(0.0, TWO_PI, size=k))
        if k > 1 and np.diff(phi).min(initial=TWO_PI) < 1e-3:
            phi += np.linspace(0.0, 1e-2, k)  # keep gaps off degeneracy
        P = 0.9 * np.stack([np.cos(phi), np.sin(phi)], axis=1)
        res0 = canonicalize_2d(P, assume_centered=True)
        for _ in range(20):
            Q = P @ rot(rng.uniform(0.0, TWO_PI)).T
            res = canonicalize_2d(Q, assume_centered=True)
            assert set_dist(res0.points, res.points) < 1e-9, k


def test_naive_counterexample():
    """Gap string (30,100,30,200): minimal gap 30 occurs twice at starts that
    are NOT related by any symmetry (the cloud is C_1). The naive rule picks
    'the maximal index', but indices are assigned by the y-axis cut of the
    input pose: rotating the cloud by 250 deg clockwise moves the cut so the
    OTHER 30-gap point becomes maximal-index -> different canonical pose."""
    P = ring_from_cw_angles([0.0, 30.0, 130.0, 160.0])  # gaps 30,100,30,200
    Q = P @ rot(np.deg2rad(-250.0)).T                    # 250 deg clockwise

    a = canonicalize_2d_naive(P, assume_centered=True)
    b = canonicalize_2d_naive(Q, assume_centered=True)
    assert set_dist(a, b) > 0.5, "naive rule unexpectedly agreed"

    ra = canonicalize_2d(P, assume_centered=True)
    rb = canonicalize_2d(Q, assume_centered=True)
    assert ra.rotational_order == 1
    assert set_dist(ra.points, rb.points) < 1e-9, "fixed rule must agree"

    # sweep: the naive rule must fail for SOME rotation, the fixed for NONE
    rng = np.random.default_rng(4)
    naive_fail = fixed_fail = 0
    for _ in range(200):
        R = rot(rng.uniform(0.0, TWO_PI))
        naive_fail += set_dist(a, canonicalize_2d_naive(P @ R.T,
                                                        assume_centered=True)) > 1e-6
        fixed_fail += set_dist(
            ra.points, canonicalize_2d(P @ R.T, assume_centered=True).points
        ) > 1e-9
    assert naive_fail > 0, "sweep failed to expose the naive bug"
    assert fixed_fail == 0, f"fixed rule failed {fixed_fail}/200 rotations"
    print(f"    (naive rule disagreed on {naive_fail}/200 random rotations; "
          f"fixed rule on 0/200)")


def test_naive_random_choice_is_set_level_benign_but_not_a_function():
    # all gaps equal (regular polygon): every choice is symmetry-related, so
    # the naive random pick still yields the same SET ...
    P = ring_from_cw_angles([0.0, 90.0, 180.0, 270.0])
    a = canonicalize_2d_naive(P, seed=0, assume_centered=True)
    b = canonicalize_2d_naive(P, seed=12345, assume_centered=True)
    assert set_dist(a, b) < 1e-9
    # ... but a canonicalization must be a deterministic FUNCTION; the fixed
    # algorithm is (bitwise identical outputs, no randomness anywhere)
    r1 = canonicalize_2d(P, assume_centered=True)
    r2 = canonicalize_2d(P, assume_centered=True)
    assert np.array_equal(r1.points, r2.points)
    assert np.array_equal(r1.order, r2.order)


def test_centroid_vs_circumcenter_hole():
    # scalene triangle inscribed in the unit circle: centroid != circumcenter,
    # so after centering the radii are NOT equal -> must refuse loudly
    P = ring_from_cw_angles([10.0, 80.0, 300.0])
    try:
        canonicalize_2d(P)  # centroid mode
        raise AssertionError("expected ValueError for non-concentric cloud")
    except ValueError:
        pass
    # about the known center it is a valid ring
    res = canonicalize_2d(P, assume_centered=True)
    assert res.rotational_order == 1


def test_edge_cases():
    # n = 1: centering sends the point to the origin (rotation acts trivially)
    res = canonicalize_2d(np.array([[3.0, 4.0]]))
    assert np.allclose(res.points, 0.0)
    # n = 1 about the origin: the point must land on the +y ray
    res = canonicalize_2d(np.array([[2.0, 0.0]]), assume_centered=True)
    assert np.allclose(res.points, [[0.0, 2.0]], atol=1e-12)
    # n = 2 centered ring: forced antipodal; canonical = {(0,r),(0,-r)}
    p = np.array([0.6, -0.8])
    res = canonicalize_2d(np.stack([p, -p]))
    assert set_dist(res.points, np.array([[0.0, 1.0], [0.0, -1.0]])) < 1e-9
    # coincident points (zero gaps) on a ring about the origin
    tri = ring_from_cw_angles([0.0, 0.0, 120.0, 240.0])  # doubled vertex
    r0 = canonicalize_2d(tri, assume_centered=True)
    rng = np.random.default_rng(5)
    for _ in range(20):
        Q = tri @ rot(rng.uniform(0.0, TWO_PI)).T
        assert set_dist(r0.points,
                        canonicalize_2d(Q, assume_centered=True).points) < 1e-9
    # all points identical: everything at the centroid; stabilizer is all of
    # SO(2), reported with the sentinel rotational_order == 0
    res = canonicalize_2d(np.full((3, 2), 5.0))
    assert np.allclose(res.points, 0.0) and res.rotational_order == 0
    # malformed input
    for bad in (np.zeros((0, 2)), np.zeros(3), np.zeros((3, 3))):
        try:
            canonicalize_2d(bad)
            raise AssertionError(f"expected ValueError for shape {bad.shape}")
        except ValueError:
            pass


def test_tiny_radius_ring_not_degenerate():
    """Regression (rigor review): the degenerate 'all at centroid' branch must
    trigger relative to the input's float-precision floor, never on an
    absolute threshold. A clean radius-1e-7 square is a perfectly valid ring
    and must canonicalize invariantly, not be returned unrotated."""
    rng = np.random.default_rng(8)
    for radius in (1e-7, 1e-12, 1e3):
        P = ring_from_cw_angles([0.0, 90.0, 180.0, 270.0], radius=radius)
        res0 = canonicalize_2d(P)
        assert res0.rotational_order == 4, radius
        p0 = res0.points[res0.order[0]]
        assert abs(p0[0]) < 1e-9 * radius and p0[1] > 0, radius
        for _ in range(20):
            Q = P @ rot(rng.uniform(0.0, TWO_PI)).T
            res = canonicalize_2d(Q)
            assert set_dist(res0.points, res.points) < 1e-9 * radius, radius


def test_radius_spread_within_tolerance():
    """Regression (rigor review): exactly-C4 ANGLES but radii spread 3e-7
    (passes the rtol=1e-6 validator). A gap-only tie-break is radius-blind and
    picks pose-dependent starts (error = the radius spread); the radius-aware
    tokens must keep the choice pose-independent and report the true C_1."""
    radii = np.array([1.0, 1.0 + 1e-7, 1.0 + 2e-7, 1.0 + 3e-7])
    P = radii[:, None] * ring_from_cw_angles([0.0, 90.0, 180.0, 270.0])
    res0 = canonicalize_2d(P, assume_centered=True)
    assert res0.rotational_order == 1, res0.rotational_order
    rng = np.random.default_rng(9)
    for _ in range(100):
        Q = P @ rot(rng.uniform(0.0, TWO_PI)).T
        res = canonicalize_2d(Q, assume_centered=True)
        assert set_dist(res0.points, res.points) < 1e-9


def test_coincident_points_rotational_order():
    """Regression (rigor review): n coincident points at nonzero radius have
    TRIVIAL stabilizer C_1 (any nonzero rotation moves them), not C_n; and a
    doubled equilateral triangle keeps its true C_3."""
    P = np.tile([[0.0, 1.0]], (6, 1))
    res = canonicalize_2d(P, assume_centered=True)
    assert res.rotational_order == 1, res.rotational_order
    assert set_dist(res.points, P) < 1e-12  # already canonical: on +y ray
    tri2 = ring_from_cw_angles([0.0, 0.0, 120.0, 120.0, 240.0, 240.0])
    res = canonicalize_2d(tri2, assume_centered=True)
    assert res.rotational_order == 3, res.rotational_order


def test_near_symmetric_stability():
    """Perturbed square. Sub-quantum perturbation (1e-12 rad << 1e-9): the
    quantized strings tie, ties resolve to symmetry-related corners, output
    stable to O(perturbation). Super-quantum (1e-6 >> 1e-9): quantization is
    faithful, the lex-min start is unique -> exact invariance."""
    rng = np.random.default_rng(6)
    for delta_deg, tol in ((np.rad2deg(1e-12), 1e-8),
                           (np.rad2deg(1e-6), 1e-9)):
        P = ring_from_cw_angles([0.0, 90.0 + delta_deg, 180.0, 270.0])
        res0 = canonicalize_2d(P, assume_centered=True)
        for _ in range(50):
            Q = P @ rot(rng.uniform(0.0, TWO_PI)).T
            res = canonicalize_2d(Q, assume_centered=True)
            assert set_dist(res0.points, res.points) < tol, delta_deg


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
