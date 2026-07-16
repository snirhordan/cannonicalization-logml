"""Tests for canonicalize_pca_degenerate: the PCA one-double/one-simple
eigenvalue stratum (>=3-fold axis: C3h/C4h/C6h/D_nh). Covers source-level
O(3) x S_n (+ translation) invariance, the eigenframe residual O(2) x O(1)
seam directly (the headline property), the staggered-ring trap that a
naive deg-plane-only baseline falls into, planar/axis-sign ties, near-
degenerate stability, scale/chirality/representativeness/determinism, off-
stratum rejection and edge cases, and a cross-check against canonicalize_3d
regime R2 on shared configurations.

Run:  python3 test_canonicalize_pca_degenerate.py   (no pytest needed)
"""

from __future__ import annotations

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'other_code'))

import numpy as np
from scipy.stats import ortho_group

from canonicalize_pca_degenerate import (
    canonicalize_pca_degenerate, canonicalize_in_eigenframe, CanonPCADegenResult)

TWO_PI = 2 * np.pi


def ring(k, r=1.0, z=0.0, phase=0.0):
    a = TWO_PI * np.arange(k) / k + phase
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.full(k, float(z))])


def pairwise(X):
    D = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    return np.sort(D[np.triu_indices(len(X), k=1)])


def rand_O3(rng, reflect=None):
    Q = ortho_group.rvs(3, random_state=rng)
    if reflect is True and np.linalg.det(Q) > 0: Q[:, 0] *= -1
    if reflect is False and np.linalg.det(Q) < 0: Q[:, 0] *= -1
    return Q


def rand_O2(rng):
    th = rng.uniform(0, TWO_PI); c, s = np.cos(th), np.sin(th)
    O = np.array([[c, -s], [s, c]])
    if rng.integers(2): O = O @ np.array([[1, 0], [0, -1]])
    return O


def residual_frame(V, deg_cols, simple_col, rng):
    Q = np.eye(3); O = rand_O2(rng)
    for a, i in enumerate(deg_cols):
        for b, j in enumerate(deg_cols):
            Q[i, j] = O[a, b]
    Q[simple_col, simple_col] = -1.0 if rng.integers(2) else 1.0
    return V @ Q


def eigframe(X):
    Xc = X - X.mean(0); lam, V = np.linalg.eigh(Xc.T @ Xc); return Xc, lam, V


def source_invariant(X, rng, trials=40, translate=True, tol_rel=1e-6):
    base = canonicalize_pca_degenerate(X).points
    scale = max(np.linalg.norm(X - X.mean(0), axis=1).max(), 1e-300)
    worst = 0.0
    for _ in range(trials):
        Q = rand_O3(rng); t = rng.uniform(-3, 3, 3) if translate else np.zeros(3)
        Xg = X[rng.permutation(len(X))] @ Q.T + t
        worst = max(worst, np.abs(canonicalize_pca_degenerate(Xg).points - base).max())
    assert worst < tol_rel * scale, (worst, scale)
    return base


# ==========================================================================
# 1) source-level invariance on rings / stacks / axial-point clouds
# ==========================================================================

def test_source_invariance_rings():
    rng = np.random.default_rng(101)
    for k in (3, 4, 5, 6):
        source_invariant(ring(k), rng)


def test_source_invariance_hex_plus_centre():
    rng = np.random.default_rng(102)
    X = np.vstack([ring(6), [[0.0, 0.0, 0.0]]])
    source_invariant(X, rng)


def test_source_invariance_stacked_rings():
    rng = np.random.default_rng(103)
    configs = [
        np.vstack([ring(3, r=1.0, z=0.5), ring(3, r=1.8, z=-0.3, phase=0.4)]),
        np.vstack([ring(4, r=1.0, z=0.6), ring(4, r=2.2, z=-0.4, phase=0.15)]),
        np.vstack([ring(6, r=1.0, z=0.3), ring(6, r=2.5, z=-0.3, phase=np.pi / 6)]),
    ]
    for X in configs:
        source_invariant(X, rng)


def test_source_invariance_axial_points_on_axis():
    rng = np.random.default_rng(104)
    X = np.vstack([ring(4, r=1.0, z=0.2), [[0.0, 0.0, 0.9], [0.0, 0.0, -1.1]]])
    source_invariant(X, rng)


def test_source_invariance_reflection_poses():
    # rand_O3(rng, reflect=True) forces det = -1 every draw -- exercise the
    # improper (mirror) part of O(3) explicitly, not just whatever mix a
    # plain rand_O3(rng) happens to sample.
    rng = np.random.default_rng(105)
    configs = [ring(3), ring(4), ring(6),
               np.vstack([ring(3, r=1.0, z=0.5), ring(3, r=1.8, z=-0.3, phase=0.4)])]
    for X in configs:
        base = canonicalize_pca_degenerate(X).points
        scale = max(np.linalg.norm(X - X.mean(0), axis=1).max(), 1e-300)
        for _ in range(20):
            Q = rand_O3(rng, reflect=True)
            assert np.linalg.det(Q) < 0
            t = rng.uniform(-3, 3, 3)
            Xg = X[rng.permutation(len(X))] @ Q.T + t
            out = canonicalize_pca_degenerate(Xg).points
            assert np.abs(out - base).max() < 1e-6 * scale


# ==========================================================================
# 2) THE HEADLINE TEST: canonicalize_in_eigenframe is a pure function of the
#    degenerate 2-PLANE + simple line, not of the particular V realizing them
# ==========================================================================

def _check_residual_invariance(X, rng, trials=30):
    Xc, lam, V = eigframe(X)
    base = canonicalize_in_eigenframe(Xc, lam, V)
    scale = max(np.linalg.norm(Xc, axis=1).max(), 1e-300)
    for _ in range(trials):
        V2 = residual_frame(V, base.deg_cols, base.simple_col, rng)
        out = canonicalize_in_eigenframe(Xc, lam, V2)
        assert np.abs(out.points - base.points).max() < 1e-8 * scale
        assert out.deg_cols == base.deg_cols and out.simple_col == base.simple_col
    return base.deg_cols


def test_eigframe_o2o1_axis_large():
    # tall narrow coaxial ring stacks: axis variance >> in-plane variance
    # -> the simple eigenvalue is the TOP one, deg_cols = (0, 1).
    rng = np.random.default_rng(201)
    configs = [
        np.vstack([ring(3, r=0.2, z=5.0), ring(3, r=0.2, z=-5.0, phase=0.5)]),
        np.vstack([ring(4, r=0.3, z=4.0), ring(4, r=0.5, z=-3.0, phase=0.2)]),
        np.vstack([ring(6, r=0.4, z=6.0), ring(6, r=0.4, z=-6.0)]),
    ]
    seen = set()
    for X in configs:
        seen.add(_check_residual_invariance(X, rng))
    assert (0, 1) in seen, seen


def test_eigframe_o2o1_axis_small():
    # flat pancake rings: axis variance << in-plane variance -> the simple
    # eigenvalue is the BOTTOM one, deg_cols = (1, 2).
    rng = np.random.default_rng(202)
    configs = [
        ring(4, r=2.0, z=0.02),
        ring(3, r=1.5, z=-0.03),
        np.vstack([ring(6, r=1.0, z=0.05), ring(6, r=2.0, z=-0.02, phase=np.pi / 6)]),
    ]
    seen = set()
    for X in configs:
        seen.add(_check_residual_invariance(X, rng))
    assert (1, 2) in seen, seen


def test_eigframe_o2o1_axial_points_mixed():
    # a few points sitting exactly (or near-exactly) on the axis, mixed with
    # non-axial ring points, in both orderings.
    rng = np.random.default_rng(203)
    configs = [
        np.vstack([ring(4, r=2.0, z=0.02), [[0.0, 0.0, 1.0], [0.0, 0.0, -1.3]]]),
        np.vstack([ring(3, r=0.15, z=5.0), ring(3, r=0.15, z=-5.0, phase=0.4),
                  [[0.0, 0.0, 0.0]]]),
    ]
    for X in configs:
        _check_residual_invariance(X, rng)


# ==========================================================================
# 3) staggered-ring trap: the real algorithm resolves it, a naive
#    deg-plane-only baseline provably does not
# ==========================================================================

def _naive_deg_only_canonicalize(X, gap_rtol=1e-6):
    """Self-contained baseline (does not reuse module internals): projects
    onto the degenerate eigenplane, sorts by in-plane angle, rotates the
    first point onto +y, attaches the axis column in that row order, and
    fixes the axis sign from the first point's height. This is exactly the
    tempting-but-wrong construction the module docstring's STAGGERED-RING
    TRAP section warns against."""
    Xc = X - X.mean(0)
    lam, V = np.linalg.eigh(Xc.T @ Xc)
    top = lam[-1]
    if (lam[1] - lam[0]) <= gap_rtol * top:
        deg_cols, simple_col = (0, 1), 2
    else:
        deg_cols, simple_col = (1, 2), 0
    Y = Xc @ V
    B = Y[:, deg_cols]
    axis = Y[:, simple_col]
    angle = np.arctan2(B[:, 1], B[:, 0])
    order = np.argsort(angle)                    # sort by in-plane angle ONLY
    B_s, axis_s, angle_s = B[order], axis[order], angle[order]
    theta = np.pi / 2.0 - angle_s[0]              # rotate first point onto +y
    c, s = np.cos(theta), np.sin(theta)
    R2 = np.array([[c, -s], [s, c]])
    B_rot = B_s @ R2.T
    sign = 1.0 if axis_s[0] >= 0.0 else -1.0       # pick axis sign from point 0
    return np.column_stack([B_rot, axis_s * sign])


def _staggered_ring(k):
    # Three coaxial k-rings at the SAME radius, offset by (2*pi/k)/3 each, at
    # generic non-symmetric heights. In-plane this is a regular 3k-gon, whose
    # 2D shadow has apparent rotational order 3k (C_3k) -- but the true 3D
    # symmetry about the axis is only C_k. The shadow/true ratio is 3.
    #
    # Why 3 and not 2: a two-ring +/-h half-step stack (ratio 2) does NOT
    # expose the naive baseline, because equal-size rings become +/-symmetric
    # after centering and the axis flip eps=-1 absorbs the extra factor of 2 as
    # a genuine improper (S_2k) symmetry -- the naive recipe stays pose-
    # invariant there. A single sign cannot absorb a factor of 3, so the
    # spurious C_3k of the shadow is real for the 2D projection yet NOT liftable
    # to any element of O(2) x O(1). This is exactly the staggered-ring trap the
    # module docstring warns about.
    step = (2.0 * np.pi / k) / 3.0
    heights = (0.3, -0.7, 1.1)
    return np.vstack([ring(k, r=1.0, z=h, phase=j * step)
                      for j, h in enumerate(heights)])


def _staggered_spurious_check(k, seed):
    rng = np.random.default_rng(seed)
    X = _staggered_ring(k)
    # (1) the REAL module is exactly pose-invariant on the staggered cloud, and
    #     its height-aware detector reports the TRUE order k, not the shadow's 3k
    source_invariant(X, rng)
    assert canonicalize_pca_degenerate(X).rotational_order == k, \
        (k, canonicalize_pca_degenerate(X).rotational_order)
    # (2) the naive isolated-2-block baseline is NOT pose-invariant here: it
    #     fixes the O(2) frame from the 3k-gon shadow alone (3x more apparent
    #     symmetry than the cloud), so different poses select genuinely different,
    #     non-symmetry-related frames -- a macroscopic O(scale) discrepancy.
    scale = max(np.linalg.norm(X - X.mean(0), axis=1).max(), 1e-300)
    outs = []
    for _ in range(8):
        Q = rand_O3(rng)
        Xg = X[rng.permutation(len(X))] @ Q.T + rng.uniform(-3, 3, 3)
        outs.append(_naive_deg_only_canonicalize(Xg))
    maxdiff = max((np.abs(outs[i] - outs[j]).max()
                   for i in range(len(outs)) for j in range(i + 1, len(outs))),
                  default=0.0)
    assert maxdiff > 1e-2 * scale, (k, maxdiff, scale)


def test_staggered_spurious_c3():
    _staggered_spurious_check(3, 300)


def test_staggered_spurious_c4():
    _staggered_spurious_check(4, 301)


def test_staggered_spurious_c6():
    _staggered_spurious_check(6, 302)


# ==========================================================================
# 4) planar clouds and the harmless sigma_h / axis-sign tie
# ==========================================================================

def _planar_axis_reversal_check(X, rng):
    Xc, lam, V = eigframe(X)
    base = canonicalize_in_eigenframe(Xc, lam, V)
    assert np.abs(Xc[:, 2]).max() < 1e-10                # every atom height ~0
    assert abs(lam[base.simple_col]) < 1e-8 * lam[-1]     # simple eigenvalue ~0
    r1 = canonicalize_pca_degenerate(X)
    r2 = canonicalize_pca_degenerate(X * np.array([1.0, 1.0, -1.0]))
    np.testing.assert_allclose(r1.points, r2.points, atol=1e-8)
    source_invariant(X, rng)


def test_planar_axis_reversal_triangle():
    rng = np.random.default_rng(400)
    X = np.vstack([[[0.0, 0.0, 0.0]], ring(3), ring(3, r=1.7, phase=0.3)])
    _planar_axis_reversal_check(X, rng)


def test_planar_axis_reversal_square():
    rng = np.random.default_rng(401)
    X = np.vstack([ring(4), ring(4, r=2.0, phase=0.2)])
    _planar_axis_reversal_check(X, rng)


def test_planar_axis_reversal_nonplanar_dnh():
    # NON-planar case: coaxial rings at +h/-h, SAME phase and radius, so
    # sigma_h (z -> -z) maps the point set to itself -- a genuine
    # self-symmetry sending the axis w to -w. Per the module docstring this
    # makes the axis SIGN unfixable by any per-point heuristic, but harmless:
    # both signs must produce the same canonical output.
    rng = np.random.default_rng(402)
    for k, h in ((3, 0.7), (4, 0.5), (6, 0.4)):
        X = np.vstack([ring(k, r=1.0, z=h), ring(k, r=1.0, z=-h)])
        source_invariant(X, rng)


# ==========================================================================
# 5) near-degenerate stability: perturbed-but-still-detected-degenerate
#    clouds must canonicalize close to the exactly-degenerate answer
# ==========================================================================

def _near_degenerate_stability_check(X0, seed):
    """Near-degeneracy (DFT-relaxed analog) stability. X0 is EXACTLY degenerate;
    X_eps = X0 + eps*noise splits the double eigenvalue by ~eps. Two DIFFERENT
    things must be distinguished (conclusions.md #3):

      * The canonical SHAPE (pairwise-distance spectrum) is Davis-Kahan STABLE:
        the degenerate-branch output depends only on the eigenPLANE, whose
        stability is governed by the O(1) gap to the SIMPLE eigenvalue, not by
        the vanishing internal split -- so the shape drifts only O(eps).
      * The canonical POSE is NOT continuous and is not asserted to be: an exact
        O(3) canonicalization is provably discontinuous at symmetric configs
        (impossibility-continuous-canonicalization), so once eps breaks the
        point-group tie a different candidate can win the argmin and the pose
        jumps by O(scale). That frame flip is expected and unavoidable; it lands
        on a different but equally valid representative (same shape).

    What IS guaranteed and asserted: (1) exact pose-invariance on each FIXED
    near-degenerate cloud, and (2) O(eps) shape stability."""
    rng = np.random.default_rng(seed)
    base_pw = pairwise(canonicalize_pca_degenerate(X0).points)
    scale = max(np.linalg.norm(X0 - X0.mean(0), axis=1).max(), 1e-300)
    for eps in (1e-10, 1e-8, 1e-6):
        X_eps = X0 + eps * rng.standard_normal(X0.shape)
        # gap_rtol relaxed to 1e-3 (vs. 1e-6 default) so the induced ~eps split
        # still classifies X_eps on the degenerate stratum (does not raise).
        res = canonicalize_pca_degenerate(X_eps, gap_rtol=1e-3)
        # (1) exact invariance still holds at this fixed, near-degenerate cloud
        b2, worst_self = res.points, 0.0
        for _ in range(20):
            Q = rand_O3(rng)
            Xg = X_eps[rng.permutation(len(X_eps))] @ Q.T + rng.uniform(-2, 2, 3)
            worst_self = max(worst_self, np.abs(
                canonicalize_pca_degenerate(Xg, gap_rtol=1e-3).points - b2).max())
        assert worst_self < 1e-5 * scale, (eps, "self-invariance", worst_self)
        # (2) Davis-Kahan shape stability: pairwise spectrum drifts only O(eps)
        shape_drift = np.abs(pairwise(res.points) - base_pw).max()
        assert shape_drift < 50.0 * eps * scale + 1e-9, (eps, "shape", shape_drift)
        # (3) pose may flip (diagnostic only) -- bounded by the cloud diameter
        assert np.abs(res.points).max() <= 2.05 * scale + 1e-9


def test_near_degenerate_stability_ring():
    _near_degenerate_stability_check(ring(4, r=1.3, z=0.0), 500)


def test_near_degenerate_stability_stacked():
    X0 = np.vstack([ring(3, r=1.0, z=0.5), ring(3, r=1.6, z=-0.4, phase=0.3)])
    _near_degenerate_stability_check(X0, 501)


# ==========================================================================
# 6) nested scales, chirality, representativeness, determinism
# ==========================================================================

def _scale_invariance_check(X0, rng, scales=(1e-8, 1.0, 1e6), trials=20):
    for scl in scales:
        X = X0 * scl
        base = canonicalize_pca_degenerate(X).points
        for _ in range(trials):
            Q = rand_O3(rng)
            out = canonicalize_pca_degenerate(X[rng.permutation(len(X))] @ Q.T).points
            assert np.abs(out - base).max() < 1e-6 * scl


def test_nested_scales_chirality_axis_large():
    rng = np.random.default_rng(600)
    X0 = np.vstack([ring(3, r=0.2, z=4.0), ring(3, r=0.3, z=-3.0, phase=0.4)])
    _scale_invariance_check(X0, rng)


def test_nested_scales_chirality_axis_small():
    rng = np.random.default_rng(601)
    X0 = np.vstack([ring(4, r=1.0, z=0.05), ring(4, r=2.0, z=-0.03, phase=0.2)])
    _scale_invariance_check(X0, rng)


def test_nested_scales_chirality_mirror():
    # a twisted stacked-triangle cloud at a GENERIC relative phase (0.7 rad,
    # neither eclipsed nor the pi/k staggered offset) is genuinely chiral --
    # its mirror image is not superimposable by rotation alone -- yet the
    # canonicalization must return identical points for the cloud and its
    # mirror image (the argmin ranges over all of O(3), improper included).
    X = np.vstack([ring(3, r=1.0, z=0.6, phase=0.0),
                   ring(3, r=1.0, z=-0.6, phase=0.7)])
    M = X * np.array([1.0, 1.0, -1.0])
    r1 = canonicalize_pca_degenerate(X)
    r2 = canonicalize_pca_degenerate(M)
    np.testing.assert_allclose(r1.points, r2.points, atol=1e-8)


def test_nested_scales_chirality_representative_and_deterministic():
    configs = [
        ring(4, r=1.3),
        np.vstack([ring(3, r=1.0, z=0.5), ring(3, r=1.8, z=-0.3, phase=0.4)]),
        np.vstack([ring(6, r=0.3, z=5.0), ring(6, r=0.3, z=-5.0)]),
    ]
    for X in configs:
        Xc = X - X.mean(0)
        res = canonicalize_pca_degenerate(X)
        np.testing.assert_allclose(pairwise(Xc), pairwise(res.points), atol=1e-8)
        assert np.abs(res.points.mean(axis=0)).max() < 1e-8
        assert sorted(res.order.tolist()) == list(range(len(X)))
        a = canonicalize_pca_degenerate(X)
        b = canonicalize_pca_degenerate(X)
        assert np.array_equal(a.points, b.points) and np.array_equal(a.order, b.order)


# ==========================================================================
# 7) rejection off-stratum, and edge cases
# ==========================================================================

def _tetrahedron(r=1.0):
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    return r * V / np.sqrt(3)


def _octahedron(r=1.0):
    return r * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], float)


def _cube(r=1.0):
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                  for sz in (-1, 1)], float)
    return r * V / np.sqrt(3)


def test_rejection_and_edges_distinct_spectrum():
    rng = np.random.default_rng(700)
    for n in (5, 12, 30):
        X = rng.normal(size=(n, 3))
        try:
            canonicalize_pca_degenerate(X)
            raise AssertionError("expected ValueError for a generic random cloud")
        except ValueError:
            pass


def test_rejection_and_edges_triple_degenerate():
    for X in (_tetrahedron(), _octahedron(), _cube()):
        try:
            canonicalize_pca_degenerate(X)
            raise AssertionError("expected ValueError for a triple-degenerate cloud")
        except ValueError:
            pass


def test_rejection_and_edges_malformed_inputs():
    for bad in (np.zeros((0, 3)), np.zeros(3), np.zeros((5, 2))):
        try:
            canonicalize_pca_degenerate(bad)
            raise AssertionError(f"expected ValueError for shape {bad.shape}")
        except ValueError:
            pass


def test_rejection_and_edges_r0_all_at_centroid():
    X = np.full((6, 3), 3.5)
    res = canonicalize_pca_degenerate(X)
    np.testing.assert_allclose(res.points, 0.0)
    assert res.rotational_order == 0 and res.n_candidates == 0


def test_rejection_and_edges_tiny_tetrahedron_not_zeroed():
    # mirrors canonicalize_3d's day-3 MAJOR regression guard: a tetrahedron
    # at radius 1e-15 about the origin is well-resolved relative to its OWN
    # scale and must not silently collapse to the R0 all-zeros branch. On
    # the triple-degenerate stratum this module's documented contract is to
    # RAISE (route to canonicalize_3d regime R3) rather than handle it --
    # lock that actual behavior explicitly.
    for r in (1e-15, 1e-16):
        try:
            canonicalize_pca_degenerate(_tetrahedron(r))
            raise AssertionError(
                "expected ValueError (triple-degenerate) for tiny tetrahedron")
        except ValueError as e:
            assert "triple" in str(e) or "spherical" in str(e), str(e)


def test_rejection_and_edges_axial_points_on_ring():
    rng = np.random.default_rng(701)
    X = np.vstack([ring(5, r=1.2, z=0.3), [[0.0, 0.0, 1.5], [0.0, 0.0, -1.7]]])
    source_invariant(X, rng)


def test_rejection_and_edges_collinear_along_axis():
    # a cloud collinear along a fixed direction: the covariance is rank 1,
    # so the perpendicular plane carries an EXACT (zero) double eigenvalue
    # and the line itself is the simple (largest) eigenvalue -- exactly the
    # one-double/one-simple stratum, with the "no non-axial point" branch of
    # _deg_candidates. Lock that this does not raise and stays invariant.
    rng = np.random.default_rng(702)
    t = np.array([-2.0, -1.0, 0.5, 1.5, 2.5])[:, None]
    X = t * np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3)
    res0 = canonicalize_pca_degenerate(X)
    assert res0.deg_cols == (0, 1) and res0.simple_col == 2
    assert res0.rotational_order == 0          # continuum stabilizer, not C_k
    source_invariant(X, rng)


# ==========================================================================
# 8) cross-check against canonicalize_3d regime R2: same orbit, not
#    (necessarily) the same matrix -- candidate generation differs
# ==========================================================================

def test_crosscheck_r2_planar_and_stacked():
    from canonicalize_3d import canonicalize_3d
    configs = [
        ring(6),
        np.vstack([[[0.0, 0.0, 0.0]], ring(3), ring(3, r=1.7, phase=0.3)]),
        np.vstack([ring(4, r=1.0, z=0.3), ring(4, r=2.5, z=-0.3, phase=0.2)]),
        np.vstack([ring(3, r=0.2, z=5.0), ring(3, r=0.3, z=-4.0, phase=0.4)]),
    ]
    for X in configs:
        r3d = canonicalize_3d(X)
        assert r3d.regime == "R2", (X.shape, r3d.regime)
        rdeg = canonicalize_pca_degenerate(X)
        np.testing.assert_allclose(pairwise(rdeg.points), pairwise(r3d.points),
                                   atol=1e-6)


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
