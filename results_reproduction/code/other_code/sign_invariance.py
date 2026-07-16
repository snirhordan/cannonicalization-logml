"""Canonicalizations of point clouds: PCA + sign fixing (any d), polar (2D).

Day-2 REVISED implementation. The original student version is preserved in git
history (commit 9920b4b); this revision fixes every finding of the day-2 review
(wiki/notes/polar-2d-canonicalization.md, wiki/notes/rigor-pass.md):

canonical_polar (findings F1-F7):
  F1  whole-cloud collapse: origin threshold is now RELATIVE to the cloud's
      max radius, so only the exactly-degenerate cloud short-circuits (and it
      returns zeros, which is pose-invariant).
  F2  tol-window H1 regression: all tie logic now runs on QUANTIZED integer
      tokens with exact comparisons; the reference point is Booth's
      lexicographically least cyclic rotation of the token string, so
      surviving ties are exact (quantized) symmetries, never index choices.
  F3  absolute radius tolerance: radii are quantized relative to the max
      radius (scale-free); angles were always scale-free.
  F4  origin-block row order: origin points quantize to the zero row; they
      are emitted first, and any residual order ambiguity is sub-quantum
      (documented) rather than macroscopic.
  F5  convention asymmetry: ONE token order everywhere - (radius, gap),
      radius-major - both for the start selection and for the orientation
      comparison.
  F6  O(m^2) walk: replaced by Booth's O(m) least-rotation (after the
      O(m log m) sort).
  F7  centering: the cloud is centered at its centroid by default
      (assume_centered=True skips it), and the docstring states the
      assumption ledger instead of an unconditional claim.

canonical_pca (all four failure strata):
  S1  degenerate spectrum: eigenvalue gaps are validated (relative rtol);
      a repeated eigenvalue raises ValueError - that stratum belongs to the
      combinatorial (WWV) route, see canonicalize_3d.py.
  S2  sign ties: the per-column sign rule is replaced by the joint rule
      proposed by the mentor - argmin over all 2^d sign patterns of the
      row-sorted matrix. Ties in this argmin occur iff the flip is a genuine
      self-symmetry, and then the tied candidates are equal matrices, so the
      output is unaffected. Needs ONLY distinct eigenvalues.
  S3  translations: the cloud is centered by default (undocumented breakage
      before); assume_centered=True skips it.
  S4  duplicate sort keys: all sorting/lexicographic comparisons run on
      QUANTIZED values, so exactly-tied keys stay tied under float noise
      instead of being ordered by eigh dust.
  Also: d is no longer hardcoded to 3 (any 1 <= d <= 12); n = 0 raises a
  clear ValueError; the argmin uses the lexicographically SMALLEST key
  (consistent with every other tie rule in the codebase); the dead `tol`
  parameter is replaced by `decimals` (quantization) and `gap_rtol`
  (spectrum validation), both actually used.

Assumption ledger (both functions). Exact invariance holds when quantization
is faithful: no compared quantity sits within float noise of a quantization
boundary, and distinct configurations differ by more than one quantum. Near
the boundaries the maps are discontinuous - unavoidable for any exact
canonicalization (wiki/notes/impossibility-continuous-canonicalization.md);
quantization concentrates the bad set instead of removing it.

Precision floor (day-3 attack finding, Break A - unfixable in float64): if
the input arrives translated by |t| much larger than the cloud's own scale,
the coordinates t + x have already lost ~log10(|t|/scale) digits of the
cloud's information in their float representation, BEFORE this function
runs. Centering removes t but cannot restore the digits: invariance then
holds only to ~eps * |t| absolute. For extreme scale ratios pre-center the
data yourself (or pass assume_centered=True on already-centered input).
"""

from __future__ import annotations

import itertools

import numpy as np

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------- shared bits

def _least_rotation_index(seq) -> int:
    """Booth's algorithm: smallest index of the lexicographically least cyclic
    rotation of seq. O(len(seq)). Elements need ==, < (use integer tuples)."""
    s = list(seq)
    n = len(s)
    if n == 0:
        raise ValueError("empty sequence")
    f = [-1] * (2 * n)
    k = 0
    for j in range(1, 2 * n):
        sj = s[j % n]
        i = f[j - k - 1]
        while i != -1 and sj != s[(k + i + 1) % n]:
            if sj < s[(k + i + 1) % n]:
                k = j - i - 1
            i = f[i]
        if sj != s[(k + i + 1) % n]:
            if sj < s[(k + i + 1) % n]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k % n


def _validate(X, d_expected=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"expected a nonempty (n, d) array, got shape {X.shape}")
    if d_expected is not None and X.shape[1] != d_expected:
        raise ValueError(f"expected shape (n, {d_expected}), got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("input contains non-finite coordinates")
    return X


# ------------------------------------------------------------- canonical_pca

def canonical_pca(X, decimals: int = 9, gap_rtol: float = 1e-6,
                  assume_centered: bool = False):
    """Canonicalize an (n, d) cloud under O(d) x S_n (+ translations).

    Centers at the centroid (skip with assume_centered=True), diagonalizes
    G = X^T X, requires a NON-DEGENERATE spectrum (pairwise eigenvalue gaps
    > gap_rtol * lambda_max; otherwise raises ValueError - use the
    combinatorial route on that stratum), and fixes the 2^d eigenvector sign
    ambiguity jointly: argmin over all sign patterns of the row-sorted
    quantized matrix. Rows are returned in the canonical sorted order.

    Guarantees (exact arithmetic + faithful quantization): representativeness
    (output = P X Q with Q in O(d)) and invariance c(P X Q + t) = c(X).
    Self-symmetric clouds tie in the argmin with EQUAL candidate matrices, so
    they are handled, not excluded.
    """
    X = _validate(X)
    n, d = X.shape
    if d > 12:
        raise ValueError(f"d = {d} would enumerate 2^{d} sign patterns; refuse")
    if not assume_centered:
        X = X - X.mean(axis=0)

    lam, V = np.linalg.eigh(X.T @ X)
    scale = float(lam[-1])
    if scale <= 0.0 or scale < (64 * np.finfo(float).eps * max(1.0, np.abs(X).max())) ** 2:
        return np.zeros_like(X)          # every point at the centroid
    gaps = np.diff(lam)
    if np.any(gaps <= gap_rtol * scale):
        raise ValueError(
            "degenerate covariance spectrum (relative eigenvalue gap <= "
            f"{gap_rtol:g}): the PCA frame is not well-defined on this cloud. "
            "This is the symmetric stratum - use the combinatorial "
            "canonicalization (canonicalize_3d) instead."
        )

    Y = X @ V
    ynorm = np.sqrt(scale)               # largest singular value ~ cloud scale
    best = best_key = None
    for signs in itertools.product((1.0, -1.0), repeat=d):
        Z = Y * np.array(signs)
        K = np.round(Z / ynorm * 10.0**decimals).astype(np.int64)
        order = np.lexsort(K.T[::-1])    # row-lexsort on QUANTIZED keys
        key = tuple(K[order].ravel())
        if best_key is None or key < best_key:
            best_key, best = key, Z[order]
    return best


# ----------------------------------------------------------- canonical_polar

def _orientation_pass(X, r, scale, decimals):
    """Canonicalize one orientation of a centered 2D cloud.

    Returns (rows, key): rows = rotated cloud in canonical row order (origin
    block first, then the ring points cyclically from the canonical start);
    key = the quantized token string read from the canonical start, used to
    compare the two orientations.
    """
    q = 10.0**decimals
    r_key = np.round(r / scale * q).astype(np.int64)
    origin = np.flatnonzero(r_key == 0)          # relative threshold (F1/F3)
    nonzero = np.flatnonzero(r_key > 0)
    m = nonzero.size
    # scale == 0 is handled by the caller, so m >= 1 here.

    alpha = np.mod(np.arctan2(X[nonzero, 1], X[nonzero, 0]), TWO_PI)
    two_pi_key = int(round(TWO_PI * q))
    # Branch-cut consistency: an angle that ROUNDS to the full circle sorts
    # at key 0, so its RAW value must be unwrapped to ~ -eps as well --
    # otherwise the raw gap differences below see a spurious ~2*pi jump
    # (day-3 regression: rotation by exactly 2*pi).
    alpha = np.where(np.round(alpha * q) >= two_pi_key, alpha - TWO_PI, alpha)
    a_key = np.round(alpha * q).astype(np.int64)
    # Quantized angles are used for SORTING only (they make the order of
    # same-angle stacks pose-stable). The gap string must be quantized from
    # RAW angle differences: gaps are rotation-invariant reals, so their
    # quantization is pose-stable, whereas quantizing absolute angles first
    # puts every exactly-regular configuration on a pose-dependent rounding
    # boundary (day-3 attack finding, Break B).
    idx = np.lexsort((r_key[nonzero], a_key))    # angle-major, radius tie-break
    order = nonzero[idx]
    a_sorted_raw = alpha[idx]
    rk_sorted = r_key[order]

    if m == 1:
        rel_key = np.array([two_pi_key], dtype=np.int64)
    else:
        d = np.diff(a_sorted_raw)          # ~-1e-16 possible on stack ties
        wrap = TWO_PI - (a_sorted_raw[-1] - a_sorted_raw[0])
        rel_key = np.round(np.concatenate([d, [wrap]]) * q).astype(np.int64)

    tokens = list(zip(rk_sorted.tolist(), rel_key.tolist()))   # ONE order (F5)
    b = _least_rotation_index(tokens)                          # Booth (F2/F6)

    theta = np.pi / 2.0 - a_sorted_raw[b]        # canonical start -> +y ray
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Y = X @ R.T

    ring_rows = np.roll(order, -b)
    row_order = np.concatenate([origin, ring_rows])
    key = tuple(tokens[b:] + tokens[:b])
    return Y[row_order], key


def canonical_polar(X, decimals: int = 9, assume_centered: bool = False):
    """Canonicalize an (n, 2) cloud under O(2) x S_n (+ translations).

    Centers at the centroid (skip with assume_centered=True). Points whose
    radius quantizes to zero relative to the max radius form the 'origin
    block' (no resolvable angle; emitted first - their residual order is
    sub-quantum by construction). The remaining points are sorted by angle,
    encoded as quantized (radius, gap-to-next) tokens, and the canonical
    start is Booth's least cyclic rotation of the token string; the start is
    rotated onto the +y ray. Both the cloud and its mirror image are
    canonicalized and the one with the lexicographically smaller token string
    is returned (equal strings = mirror-symmetric cloud, either works).

    Guarantees (faithful quantization): representativeness and invariance
    under rotations, reflections, row permutations, and (by centering)
    translations. Surviving ties are quantized symmetries: harmless.
    """
    X = _validate(X, d_expected=2)
    if not assume_centered:
        X = X - X.mean(axis=0)
    r = np.linalg.norm(X, axis=1)
    scale = float(r.max())
    if scale <= 64 * np.finfo(float).eps * max(1.0, np.abs(X).max()):
        return np.zeros_like(X)          # exactly/sub-noise degenerate cloud

    Y1, k1 = _orientation_pass(X, r, scale, decimals)
    Xp = X * np.array([1.0, -1.0])
    Y2, k2 = _orientation_pass(Xp, r, scale, decimals)
    if k2 < k1:
        return Y2
    return Y1
