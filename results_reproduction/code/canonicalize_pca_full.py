"""Full PCA-based canonicalization of (n,3) point clouds: DISPATCH on the
centered covariance spectrum to whichever handler is numerically trustworthy
on that stratum, and never raise.

This module does not implement any new canonicalization logic itself -- it
only classifies the spectrum of the centered Gram matrix C = X^T X (via
np.linalg.eigh, ascending eigenvalues lam) and routes to the existing,
already-reviewed handlers:

  R0 (top eigenvalue ~ 0)        -> every point at the centroid; return zeros.
  DISTINCT (no small gap)        -> other_code/sign_invariance.canonical_pca
                                     (repaired-PCA route: unique eigenframe up
                                     to column sign, argmin over the 2^3 sign
                                     patterns).
  AXIAL (exactly one small gap)  -> canonicalize_pca_degenerate.
                                     canonicalize_in_eigenframe (one-double-
                                     one-simple stratum: O(2) in-plane
                                     enumeration, stable to in-plane
                                     eigenvector rotation).
  TRIPLE (both gaps small)       -> canonicalize_3d.canonicalize_3d, which
                                     lands in its regime R3 (spherical/cubic
                                     stratum: axes come from the point set,
                                     not from the degenerate eigenframe).

WHY THE DISPATCH THRESHOLD MUST BE LOOSE (the point of this module).
canonical_pca is only numerically trustworthy when the covariance spectrum
is WELL-SEPARATED: its output is built directly from the eigenVECTORS of
X^T X, and an eigenvector belonging to a near-degenerate eigenvalue swings
by an angle of order eps / gap under an O(eps) perturbation of the input
(standard eigenvector perturbation bound, e.g. Davis-Kahan) -- so as the gap
between two eigenvalues shrinks towards float noise, canonical_pca's chosen
frame becomes numerically unstable even though its SIGN-fixing step is
already robust. canonicalize_in_eigenframe and canonicalize_3d's R3 route,
by contrast, do not trust any single eigenvector on the degenerate subspace:
they enumerate a whole family of candidate in-plane / axis directions (O(2)
rotations, or point-set-derived axes) and take a whole-cloud quantized
argmin, which is stable under exactly the eigenvector rotation that PCA's
frame is unstable to.

Consequently, the dispatch threshold `gap_rtol` (the relative-gap cutoff
used to call two eigenvalues "close") must be set LARGER than the internal
tolerance canonical_pca itself uses to decide whether to raise
(canonical_pca's own default gap_rtol is 1e-6, tuned only to catch EXACTLY
tied eigenvalues, not near-ties). If this module used that same tight
threshold, a cloud whose gap is, say, 1e-4 relative to lambda_max would be
routed to canonical_pca, which would happily return an eigenvector-based
frame that is only good to ~1e-4/1e-6 ~ 100x worse conditioning than the cloud's
own coordinate precision -- exactly the failure mode this module exists to
avoid. So DEFAULT_GAP_RTOL below is chosen on the order of 1e-3 (three
orders of magnitude looser than canonical_pca's raise threshold), and
whenever this module DOES call canonical_pca (because the dispatch has
already decided, at the loose threshold, that the spectrum is well-separated)
it passes canonical_pca a much TIGHTER gap_rtol (_PCA_INTERNAL_GAP_RTOL,
1e-9) purely so that canonical_pca does not re-raise ValueError over a gap
this module has already judged safe -- the dispatch's own gap_rtol is the
real decision; canonical_pca's internal check is defused, not repeated.

DEFAULT_GAP_RTOL is a single module-level constant precisely so Phase 2 can
retune it without touching the dispatch logic.

ORDER FOR THE DISTINCT BRANCH. canonical_pca returns only the (n,3) canonical
points, not the permutation of input row indices that produced them (unlike
canonicalize_in_eigenframe / canonicalize_3d, which both return `.order`).
To fill the `order` field of this module's uniform NamedTuple without
touching canonical_pca's internals, this module recomputes the SAME
eigendecomposition it needs anyway for the dispatch (lam, V = eigh(X^T X)),
forms Y = X_centered @ V, and matches each output row back to the row of Y
it came from using the fact that canonical_pca's only unknown transform is a
column-wise SIGN pattern applied to Y (never a row mixing): |Z[i, :]| =
|Y[i, :] * signs| = |Y[i, :]| for any sign vector in {+-1}^3, since row i
of Y is never touched by any OTHER row. So the componentwise absolute value
of each row is an exact, sign-pattern-independent fingerprint of which input
row it is, and canonical_pca only permutes rows afterwards -- it never
mixes them. A greedy nearest-neighbor match on these fingerprints (exact up
to floating-point determinism of two independent eigh calls on the same
matrix) recovers the permutation.
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple

import numpy as np

# --------------------------------------------------------------------------
# Robust imports (resolved from __file__, not from the launch CWD).

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from canonicalize_pca_degenerate import canonicalize_in_eigenframe  # noqa: E402
from canonicalize_3d import canonicalize_3d  # noqa: E402

_OTHER_CODE_DIR = os.path.join(_THIS_DIR, "other_code")
if _OTHER_CODE_DIR not in sys.path:
    sys.path.insert(0, _OTHER_CODE_DIR)
from sign_invariance import canonical_pca  # noqa: E402


# Dispatch threshold: relative-gap cutoff (vs lambda_max) at or below which
# two eigenvalues are treated as "too close to trust an eigenvector from".
# See module docstring: deliberately looser than canonical_pca's own raise
# threshold (1e-6) so near-degenerate spectra are routed to the stable
# enumeration handlers instead of to canonical_pca's unstable eigenframe.
# Phase 2 will tune this value; keep it a single named constant.
DEFAULT_GAP_RTOL = 2e-2

# gap_rtol passed to canonical_pca itself once the dispatch has ALREADY
# decided (at DEFAULT_GAP_RTOL, or whatever gap_rtol the caller supplied)
# that the spectrum is well-separated. Tiny on purpose: it only needs to
# stop canonical_pca from re-raising over floating-point dust in a gap this
# module has already judged safe.
_PCA_INTERNAL_GAP_RTOL = 1e-9


class CanonPCAFullResult(NamedTuple):
    """points: (n,3) canonical pose, rows in canonical order.
    order: permutation of input row indices realizing that row order.
    regime: 'R0' | 'DISTINCT' | 'AXIAL' | 'TRIPLE'.
    n_candidates: size of the enumerated candidate set of the handler used
        (0 for R0; 8 for DISTINCT, the fixed 2^3 sign-pattern count of
        canonical_pca; handler-reported for AXIAL / TRIPLE)."""

    points: np.ndarray
    order: np.ndarray
    regime: str
    n_candidates: int


def _validate(P):
    X = np.asarray(P, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0:
        raise ValueError(f"expected a nonempty (n, 3) array, got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("input contains non-finite coordinates")
    return X


def _match_order(fingerprint_pool: np.ndarray, fingerprints_out: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor matching (without replacement) of each row of
    `fingerprints_out` to a row of `fingerprint_pool`. Both are (n, 3)
    sign-independent row fingerprints (see module docstring); the match is
    exact (up to floating-point determinism) because canonical_pca never
    mixes rows, only permutes and column-sign-flips them.
    """
    n = fingerprint_pool.shape[0]
    remaining = list(range(n))
    order = np.empty(n, dtype=int)
    for k in range(n):
        diffs = fingerprint_pool[remaining] - fingerprints_out[k]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        j = int(np.argmin(d2))
        order[k] = remaining.pop(j)
    return order


def canonicalize_pca_full(
    P,
    decimals: int = 9,
    assume_centered: bool = False,
    gap_rtol: float = DEFAULT_GAP_RTOL,
    axial_rtol: float = 1e-6,
) -> CanonPCAFullResult:
    """Full PCA-based canonicalization of an (n,3) cloud under O(3) x S_n
    (+ translations). NEVER raises: dispatches on the centered covariance
    spectrum to whichever handler is numerically trustworthy for that
    stratum. See module docstring for the noise-robustness reasoning behind
    `gap_rtol`'s default and for how `order` is recovered on the DISTINCT
    branch (the only handler here that does not natively report it).
    """
    X = _validate(P)
    n = X.shape[0]
    input_mag = float(np.abs(X).max())
    Xc = X if assume_centered else X - X.mean(axis=0)

    lam, V = np.linalg.eigh(Xc.T @ Xc)  # ascending
    top = float(lam[-1])

    # Same noise-floor convention as canonicalize_3d / canonical_pca: judged
    # against the RAW input magnitude (centering cancellation error scales
    # with |input|, not with the already-centered scale).
    noise2 = (64.0 * np.finfo(float).eps * max(1.0, input_mag)) ** 2
    if top <= max(noise2, 0.0):
        return CanonPCAFullResult(np.zeros_like(Xc), np.arange(n), "R0", 0)

    gap01_small = (lam[1] - lam[0]) <= gap_rtol * top
    gap12_small = (lam[2] - lam[1]) <= gap_rtol * top

    if not gap01_small and not gap12_small:
        # DISTINCT: well-separated spectrum, canonical_pca's eigenframe is
        # trustworthy. Defuse its internal (much tighter) raise threshold --
        # this dispatch has already made the real decision.
        points = canonical_pca(
            Xc, decimals=decimals, gap_rtol=_PCA_INTERNAL_GAP_RTOL,
            assume_centered=True,
        )
        Y = Xc @ V
        order = _match_order(np.abs(Y), np.abs(points))
        return CanonPCAFullResult(points, order, "DISTINCT", 2 ** X.shape[1])

    if gap01_small != gap12_small:
        # AXIAL: exactly one gap small (one-double-one-simple stratum).
        res = canonicalize_in_eigenframe(
            Xc, lam, V, decimals=decimals, gap_rtol=gap_rtol,
            axial_rtol=axial_rtol,
        )
        return CanonPCAFullResult(res.points, res.order, "AXIAL",
                                  res.n_candidates)

    # TRIPLE: both gaps small (spherical/cubic stratum) -> canonicalize_3d,
    # which lands in its regime R3 (axes derived from the point set itself,
    # not from the uninformative degenerate eigenframe).
    res = canonicalize_3d(
        P, decimals=decimals, gap_rtol=gap_rtol, axial_rtol=axial_rtol,
        assume_centered=assume_centered,
    )
    return CanonPCAFullResult(res.points, res.order, "TRIPLE", res.n_candidates)
