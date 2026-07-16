"""Canonicalization of 3D point clouds on the PCA axial-degeneracy stratum:
covariance spectrum = one double eigenvalue + one simple eigenvalue (the
>=3-fold-axis stratum: C3h, C4h, C6h, D_nh (n>=3), ... -- see
../wiki/notes/conclusions.md S1 and ../wiki/notes/pca-canonicalization-flaws.md
Flaw 2). On every OTHER spectrum this module refuses (ValueError): distinct
spectrum -> other_code/sign_invariance.py:canonical_pca; triple degeneracy ->
canonicalize_3d.py regime R3.

Setup. Center X (n,3), C = X^T X = V diag(lam) V^T (eigh, lam ascending). On
this stratum lam = (l, m, m) or (m, m, l) with l != m: the eigenspace of the
double eigenvalue is a 2-PLANE, not two separate lines, so its eigenbasis is
defined only up to an arbitrary O(2) rotation/reflection; the simple
eigenvector is defined only up to sign. The residual ambiguity of the WHOLE
eigendecomposition is therefore O(2) x O(1), realized as Y = X @ V -> Y @ Q,
Q = blockdiag(O, eps) with O in O(2) on the two degenerate columns (deg_cols)
and eps in {+1,-1} on the simple column (simple_col). This is a COLUMN action
on Y: it re-expresses each row's 3 coordinates in a different (but equally
valid) eigenbasis, not a row permutation. Canonicalizing Y against this
residual, plus sorting rows against S_n, gives a canonical pose.

THE STAGGERED-RING TRAP (why the 2-block cannot be canonicalized in
isolation). Project a cloud onto the degenerate 2-plane and canonicalize that
2D projection on its own (e.g. with sign_invariance.canonical_polar) -- this
is tempting and WRONG. Two same-radius rings at different heights, offset by
a half in-plane step (e.g. a C3h molecule: one ring of 3 atoms at height +h,
another ring of 3 atoms at height -h rotated by 60 degrees) project to a
regular HEXAGON: the 2D shadow has apparent symmetry C6, but the actual 3D
cloud only has C3 about that axis (plus the horizontal mirror). Any O(2)
frame fixed from the 2D projection alone throws away the height information
that breaks 3 of the hexagon's 6 apparent symmetries; attaching the 3rd
(axis) column AFTER that choice is already too late -- invariance breaks,
because two rotations that tie in the 2D shadow generally do NOT tie once the
axis column is attached. The fix: the 2D polar machinery (arctan2 + rotate a
reference point onto the +y ray, following sign_invariance.canonical_polar /
_orientation_pass) is used ONLY to PROPOSE a finite set of candidate in-plane
orientations; which one WINS is decided by the argmin of the quantized,
row-lexsorted key of the FULL (y1, y2, y3) triple -- axis column included.
This is exactly regime R2 of canonicalize_3d.py, specialized to a fixed axis
instead of a searched one.

AXIS SIGN (eps) IS AN ARGMIN DIMENSION, NOT A HEURISTIC. The usual fix ("flip
the sign so some reference point has positive height") fails two ways here:
(1) planar C3h/C6h molecules have EVERY atom at height 0 (h identically
zero) -- there is no reference point with nonzero height to orient by; (2) a
cloud with a self-symmetry that reverses the axis (some g in the point group
sends w -> -w) makes the sign genuinely unfixable by any rule -- but this is
HARMLESS, because that symmetry maps the cloud to itself, so both signs
produce the SAME output cloud (a quantized tie). Enumerating eps in
{+1, -1} as one more axis of the argmin, rather than picking it by a
heuristic, handles both cases uniformly: case (1) is simply always a tie
(h = 0 either way); case (2) is a tie because the underlying clouds coincide.

CANDIDATE SET. In eigenframe coordinates Y = X @ V: B = Y[:, deg_cols] (the
in-plane block), a = Y[:, simple_col] (the axis column). scale = max row norm
of centered X. A point is AXIAL if its in-plane radius rho_i = |B_i| is below
axial_rtol * scale (same conditioning rule as canonicalize_3d._axial_candidates
-- normalizing a near-zero in-plane vector amplifies float error by
scale / rho and cannot define a stable in-plane direction).
  - No non-axial point (cloud collinear along the axis): the in-plane
    completion is provably immaterial (every point reads (0, 0, h) in ANY
    completion), so a single arbitrary completion is emitted per eps, and the
    key is computed from the AXIS PROJECTION ONLY (mirrors canonicalize_3d's
    collinear branch) so that poses agree up to sub-resolution dust rather
    than by an arbitrary O(scale) perpendicular jump.
  - Otherwise: for EACH non-axial point i, and each reflection flag in
    {False, True}, build the O(2) element that rotates B so point i's azimuth
    lands on the +y ray (the reflection flag flips in-plane handedness before
    rotating, so it still lands point i on +y but with det = -1); for each
    eps in {+1, -1}, Q = blockdiag(O, eps); R = V @ Q in O(3); the candidate
    key is the quantized row-lexsort of X @ R (equivalently Y @ Q). The
    GLOBAL lex-argmin over this whole set wins. Using every non-axial point
    (rather than only the smallest-quantized-radius reference ring, as
    canonicalize_3d's R2 does for speed) is a safe superset: correctness
    first, the reference-ring restriction is a valid but unimplemented
    optimization. All comparisons run on integers quantized to `decimals`
    relative to scale (exactly _key_for_frame / canonical_pca's convention),
    so float-equal ties survive as exact ties instead of being broken by
    eigh's arbitrary rounding dust.

GUARANTEES. Representativeness: the output is P X R with R in O(3) (a
composition of the fixed eigenbasis V with an element of the residual
O(2) x O(1) group) and P a row permutation -- an orbit member. Invariance:
c(P0 X R0^T + t) = c(X) for any R0 in O(3), P0 in S_n: centering removes t;
the covariance transforms as C -> R0 C R0^T, so its eigenvalues (hence the
deg_cols/simple_col split) are unchanged and an eigenbasis of the
transformed cloud is R0 V for the SAME V (up to the same O(2) x O(1)
residual) -- so the candidate SET of (O, eps) pairs, and therefore the
multiset of quantized keys, is exactly the same before and after the
transform; the argmin picks the same key, hence the same output up to the
residual's own quantized ties. Ties in the argmin are quantized symmetries:
tied candidates give byte-for-byte equal quantized keys, so the returned
`points` are unaffected (in exact arithmetic; see the float-precision note
below for generic, non-axis-aligned residual rotations). Discontinuities
exist only on the (measure-zero, unavoidable) quantization-boundary set,
exactly as documented for canonicalize_3d, canonicalize_2d and
sign_invariance's canonical_pca / canonical_polar.

Assumption ledger / float-precision note. `canonicalize_in_eigenframe` is
purely a function of the *quantized* candidate keys: replacing V by V @ Q0
for Q0 = blockdiag(O_random, eps0) in the residual group produces the exact
same SET of candidate frames R = V @ Q (a re-indexing, proved in-file), hence
the same minimal quantized key -- always. When that minimum is achieved by a
UNIQUE candidate, and O_random is itself exactly float64-representable
(permutations, sign flips, 90-degree rotations), the returned `points` are
genuinely byte-identical; for a generic (irrational-angle) O_random the two
computations reach the same matrix by different floating-point paths and can
differ at the ~1e-15 relative level even though they are mathematically the
same O(3) matrix -- the same caveat every other module in this project states
for tie survival under float noise.

Robust imports: canonicalize_3d.py lives next to this file (its dir is put on
sys.path from __file__, not relying on CWD); sign_invariance.py lives in
other_code/ (only used for Booth's _least_rotation_index, purely to pick a
canonical reporting start for the rotational_order diagnostic -- never for
correctness-critical candidate generation, which is brute-force per the
CANDIDATE SET section above).
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple

import numpy as np

# --------------------------------------------------------------------------
# Robust imports (work regardless of CWD -- resolved from __file__, not from
# whatever directory the interpreter happens to be launched in).

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from canonicalize_3d import _validate as _validate_points  # noqa: E402
from canonicalize_3d import _key_for_frame  # noqa: E402

_OTHER_CODE_DIR = os.path.join(_THIS_DIR, "other_code")
if _OTHER_CODE_DIR not in sys.path:
    sys.path.insert(0, _OTHER_CODE_DIR)
try:
    from sign_invariance import _least_rotation_index  # noqa: E402
except ImportError:                       # pragma: no cover - defensive only
    _least_rotation_index = None


class CanonPCADegenResult(NamedTuple):
    """points: (n,3) canonical pose, rows in canonical order.
    order: permutation of input row indices realizing that row order.
    deg_cols: the two columns of V spanning the degenerate eigenplane, e.g.
        (0,1) or (1,2) (lam ascending -> the double eigenvalue is always the
        two smallest or the two largest).
    simple_col: the column of V for the simple eigenvalue.
    rotational_order: detected in-plane rotational symmetry order of the
        FULL 3D cloud about the axis (0 if there is no non-axial point, i.e.
        the completion is immaterial and the true stabilizer is a continuum,
        not a finite C_k). Computed from ALL non-axial points' (radius,
        height, angular-gap) tokens -- height is included precisely so a
        staggered-ring cloud is not misreported at its projection's (larger,
        wrong) apparent order; see module docstring.
    n_candidates: size of the enumerated (O(2) x eps) candidate set."""

    points: np.ndarray
    order: np.ndarray
    deg_cols: tuple
    simple_col: int
    rotational_order: int
    n_candidates: int


def _detect_deg_split(lam, gap_rtol):
    """Classify the covariance spectrum lam (ascending) by relative gaps.

    Returns (deg_cols, simple_col) on the one-double-one-simple stratum.
    Raises ValueError on a fully non-degenerate spectrum (use canonical_pca)
    or a triple-degenerate one (use canonicalize_3d regime R3). The
    commutation lemma RCR^T = C for point-group rotations R (see
    ../wiki/notes/conclusions.md S1) is why point-group symmetry forces
    exactly this trichotomy on the spectrum of C."""
    lam = np.asarray(lam, dtype=float)
    if lam.shape != (3,):
        raise ValueError(f"expected 3 eigenvalues, got shape {lam.shape}")
    top = float(lam[-1])
    gap01_small = (lam[1] - lam[0]) <= gap_rtol * top
    gap12_small = (lam[2] - lam[1]) <= gap_rtol * top

    if not gap01_small and not gap12_small:
        raise ValueError(
            "non-degenerate covariance spectrum (all eigenvalue gaps exceed "
            f"{gap_rtol:g} * lambda_max): this cloud has no PCA degeneracy -- "
            "use canonical_pca (other_code/sign_invariance.py) instead."
        )
    if gap01_small and gap12_small:
        raise ValueError(
            "triple-degenerate covariance spectrum (lambda1 ~ lambda2 ~ "
            "lambda3): this is the spherical/cubic stratum, not the axial "
            "one -- use canonicalize_3d (regime R3) instead."
        )
    if gap01_small:
        return (0, 1), 2      # bottom two degenerate; simple eigval on top
    return (1, 2), 0           # top two degenerate; simple eigval on bottom


def _deg_candidates(Y, V, deg_cols, simple_col, scale, axial_rtol):
    """Enumerate (Q, X_for_key) candidates: Q is a 3x3 orthogonal matrix
    realizing one element of the O(2) x O(1) residual (identity outside the
    deg_cols/simple_col positions); X_for_key is None (compare on the full
    cloud) or an explicit array to key on instead (the collinear branch).
    See the CANDIDATE SET section of the module docstring."""
    B = Y[:, deg_cols]
    rho = np.linalg.norm(B, axis=1)
    nonaxial = np.flatnonzero(rho >= axial_rtol * scale)

    if nonaxial.size == 0:
        # Collinear-at-resolution along the axis: every row reads (0, 0, h)
        # under ANY in-plane completion, so the completion is immaterial and
        # must not leak dust into the key. axis_only = outer(h, axis_hat) in
        # AMBIENT coordinates, h = Y[:, simple_col] and axis_hat = V[:,
        # simple_col] (both UNSIGNED): (axis_only @ R) always works out to
        # (0, 0, eps*h) for R = V @ Q regardless of eps, because the eps sign
        # cancels between h's own definition and axis_hat's role in R -- see
        # module docstring's float-precision note and canonicalize_3d's
        # identical collinear branch (_axial_candidates).
        axis_only = np.outer(Y[:, simple_col], V[:, simple_col])
        candidates = []
        for eps in (1.0, -1.0):
            Q = np.eye(3)
            Q[simple_col, simple_col] = eps
            candidates.append((Q, axis_only))
        return candidates

    candidates = []
    for i in nonaxial:
        for reflect in (False, True):
            b = B[i] * np.array([1.0, -1.0]) if reflect else B[i]
            alpha = np.arctan2(b[1], b[0])
            theta = np.pi / 2.0 - alpha        # -> point i lands on +y ray
            c, s = np.cos(theta), np.sin(theta)
            O2 = np.array([[c, -s], [s, c]]).T
            if reflect:
                O2 = np.array([[1.0, 0.0], [0.0, -1.0]]) @ O2
            for eps in (1.0, -1.0):
                Q = np.eye(3)
                Q[np.ix_(deg_cols, deg_cols)] = O2
                Q[simple_col, simple_col] = eps
                candidates.append((Q, None))
    return candidates


def _rotational_order(Y, deg_cols, simple_col, scale, decimals, axial_rtol):
    """Diagnostic only (does not feed the argmin). Detected in-plane
    rotational order of the FULL 3D cloud about the axis, from ALL non-axial
    points sorted by azimuth and tokenized as (radius, height, angular gap)
    -- the height is what prevents the staggered-ring trap (see module
    docstring): a 2D-only gap string would read the projected hexagon's
    period, overcounting a C3h cloud as C6. Booth's least-rotation index
    (reused from sign_invariance, when importable) only picks a canonical
    reporting start; it does not affect the returned period, which is
    rotation-of-the-cyclic-string invariant by construction."""
    B = Y[:, deg_cols]
    rho = np.linalg.norm(B, axis=1)
    nonaxial = np.flatnonzero(rho >= axial_rtol * scale)
    if nonaxial.size == 0:
        return 0                              # continuum stabilizer, not C_k

    q = 10.0 ** decimals
    alpha = np.mod(np.arctan2(B[nonaxial, 1], B[nonaxial, 0]), 2.0 * np.pi)
    # Order azimuthally on the QUANTIZED angle, breaking exact-azimuth ties
    # (e.g. an eclipsed prism's +h/-h points at the same in-plane angle) by
    # quantized height then radius. A raw-angle argsort would resolve those
    # ties by sub-quantum float dust from eigh, scrambling the (rho,h,gap)
    # token sequence and collapsing the detected period to 1. Quantized keys
    # make the ordering invariant to that dust (matches sign_invariance's
    # a_key convention). [Review finding: rotational_order diagnostic.]
    a_key_all = np.round(alpha * q).astype(np.int64)
    rho_key_all = np.round(rho[nonaxial] / scale * q).astype(np.int64)
    h_key_all = np.round(Y[nonaxial, simple_col] / scale * q).astype(np.int64)
    idx = np.lexsort((rho_key_all, h_key_all, a_key_all))
    alpha_s = alpha[idx]
    rho_key = rho_key_all[idx]
    h_key = h_key_all[idx]
    gaps = np.diff(np.concatenate([alpha_s, [alpha_s[0] + 2.0 * np.pi]]))
    gap_key = np.round(gaps * q).astype(np.int64)
    tokens = list(zip(rho_key.tolist(), h_key.tolist(), gap_key.tolist()))

    m = len(tokens)
    if _least_rotation_index is not None:
        b = _least_rotation_index(tokens)     # canonical start (cosmetic)
        tokens = tokens[b:] + tokens[:b]
    for p in range(1, m + 1):                 # smallest period p | m
        if m % p:
            continue
        if all(tokens[k] == tokens[k % p] for k in range(m)):
            return m // p
    return 1                                  # unreachable: p = m always works


def canonicalize_in_eigenframe(
    X_centered,
    lam,
    V,
    decimals: int = 9,
    gap_rtol: float = 1e-6,
    axial_rtol: float = 1e-6,
) -> CanonPCADegenResult:
    """THE CORE. Takes an EXPLICIT eigenframe: lam ascending (3,), V (3,3)
    with V[:, k] the eigenvector of lam[k], and the (already centered) cloud
    X_centered whose covariance they diagonalize. Detects deg_cols/simple_col
    from lam alone (raises ValueError off-stratum, same rule as
    canonicalize_pca_degenerate), builds Y = X_centered @ V, enumerates the
    (O(2) x eps) candidate set (_deg_candidates) and returns the global
    lex-argmin over the quantized row key. See module docstring for the
    purity guarantee: calling this with V and with V @ blockdiag(O, eps) for
    ANY O in O(2) on deg_cols and eps in {+1,-1} returns the same candidate
    SET (hence the same minimal quantized key, hence the same `points` up to
    the float-precision note above).
    """
    X = _validate_points(X_centered)
    n = X.shape[0]
    lam = np.asarray(lam, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.shape != (3, 3):
        raise ValueError(f"expected V.shape == (3, 3), got {V.shape}")
    deg_cols, simple_col = _detect_deg_split(lam, gap_rtol)

    scale = float(np.linalg.norm(X, axis=1).max())
    if scale <= 0.0:
        # Defensive: a direct core caller supplying an all-zero cloud. The
        # wrapper's R0 guard normally prevents this from ever being reached.
        return CanonPCADegenResult(np.zeros_like(X), np.arange(n), deg_cols,
                                   simple_col, 0, 0)

    Y = X @ V
    candidates = _deg_candidates(Y, V, deg_cols, simple_col, scale, axial_rtol)

    best_key = best_order = best_R = None
    for Q, X_for_key in candidates:
        R = V @ Q
        source = X if X_for_key is None else X_for_key
        key, order, _ = _key_for_frame(source, R, scale, decimals)
        if best_key is None or key < best_key:
            best_key, best_order, best_R = key, order, R

    rot_order = _rotational_order(Y, deg_cols, simple_col, scale, decimals,
                                  axial_rtol)
    points = X @ best_R
    return CanonPCADegenResult(points[best_order], best_order, deg_cols,
                               simple_col, rot_order, len(candidates))


def canonicalize_pca_degenerate(
    P,
    decimals: int = 9,
    gap_rtol: float = 1e-6,
    axial_rtol: float = 1e-6,
    assume_centered: bool = False,
) -> CanonPCADegenResult:
    """Canonicalize a (n,3) cloud on the PCA one-double-one-simple stratum
    (see module docstring). Centers at the centroid (skip with
    assume_centered=True); on the R0 sub-case (every point already at the
    centroid -- the noise-floor guard is relative to the RAW input magnitude,
    matching canonicalize_3d's day-3 fix) returns zeros directly rather than
    raising, since that pose is trivially invariant. RAISES ValueError if the
    spectrum is not this stratum's: distinct (-> canonical_pca) or triple
    (-> canonicalize_3d regime R3).
    """
    X = _validate_points(P)
    n = X.shape[0]
    input_mag = float(np.abs(X).max())
    if not assume_centered:
        X = X - X.mean(axis=0)

    lam, V = np.linalg.eigh(X.T @ X)           # ascending eigenvalues
    top = float(lam[-1])
    noise2 = (64.0 * np.finfo(float).eps * input_mag) ** 2
    if top <= max(noise2, 0.0):                # R0: every point at centroid
        return CanonPCADegenResult(np.zeros_like(X), np.arange(n), (0, 1), 2,
                                   0, 0)

    _detect_deg_split(lam, gap_rtol)           # raises on the wrong stratum
    return canonicalize_in_eigenframe(X, lam, V, decimals, gap_rtol,
                                      axial_rtol)
