"""Canonicalization of 3D point clouds under O(3) x S_n (+ translations).

THE PROPOSED CANONICALIZATION is canonicalize_3d_wwv: the PCA-free 2D polar
construction (Wolter-Woo-Volz encoding + least-rotation argmin) lifted to
3D. It needs NO eigendecomposition: candidate axes come from the cloud's
own reference shell, azimuths from the reference ring about each axis, and
the canonical pose is the argmin of a quantized scale-relative total key.
See project_plan/canonicalization3d.html for the line-by-line walkthrough.

canonicalize_3d (below) is an eigenframe-ACCELERATED variant of the same
argmin principle (8 sign frames when the covariance spectrum is
non-degenerate); it shares all helpers and guarantees but is an
optimization, not the proposal.

Both are argmins over a FINITE, EQUIVARIANTLY-CONSTRUCTED set of candidate
orthogonal frames of a quantized total key, following the two 2D lessons
(wiki/notes/naive-2d-canonicalization-holes.md, polar-2d-canonicalization.md):
ties must be broken by whole-configuration comparison, and every comparison
must run on quantized values relative to the cloud's scale.

Regimes, by the spectrum of the centered Gram C = X^T X (eigenvalues
l1 <= l2 <= l3, relative gaps vs gap_rtol * l3) -- the PCA-degeneracy
dichotomy of the project (wiki/notes/conclusions.md):

R0  l3 ~ 0: every point at the centroid. Return zeros (pose-invariant).
R1  both gaps large (non-degenerate): the eigenbasis V is unique up to
    column signs; candidates = { V.S : S in {+-1}^3 } (8 frames, both
    chiralities). This is the repaired-PCA route.
R2  exactly one gap small (axial degeneracy, e.g. C3v/C4v/C6h molecules or
    planar rings): the SIMPLE eigenvalue's eigenvector w is unique up to
    sign; candidates = for each axis z in {+w, -w}: cylindrical coordinates
    about z, the reference ring = non-axial points with the lexicographically
    smallest quantized (rho, h) pair, one frame per reference-ring point
    (its azimuth -> 0) and its mirror (det = -1). If every point is axial
    (collinear cloud) the completion of the frame is immaterial -- axial
    points read (0, 0, h) in ANY completion -- so a single completion per
    axis sign suffices.
R3  both gaps small (spherical degeneracy: tetrahedral/octahedral/... or
    near-isotropic clouds): the eigenframe carries no information. Axis
    candidates come from the point set itself: the reference SHELL = points
    with the smallest positive quantized radius; each shell point's direction
    is a candidate +z axis; per axis, azimuth candidates as in R2.

Candidate counts: 8 (R1), O(ring) (R2), O(shell * ring) (R3) -- at molecular
scale (n <= 80) at most a few thousand keys of O(n log n) each. No silent
caps: the full candidate sets are always enumerated.

Guarantees. Representativeness: the output is P.X.R with R in O(3) and P a
row permutation, hence an orbit member. Invariance: every candidate-defining
object (eigenvalues, eigenvectors up to sign, radii, cylindrical (rho, h)
pairs, quantized keys) is equivariant, so the candidate FRAME SET of a
transformed cloud is the transformed frame set and the argmin key multiset is
identical; ties in the argmin mean equal quantized configurations, i.e. the
tied frames differ by a (quantized) symmetry, so the returned matrix is
unaffected. Assumption ledger: quantization faithfulness (no compared
quantity within float noise of a quantization boundary; distinct
configurations separated by more than one quantum) and regime stability
(spectral gaps not within noise of gap_rtol * l3 -- an adversarial cloud
whose gap sits EXACTLY at the threshold flips regime across poses, as the
day-3 attack round confirmed). On the boundary sets the map is
discontinuous -- unavoidable for any exact canonicalization
(wiki/notes/impossibility-continuous-canonicalization.md). Collinear-at-
resolution clouds: the frame completion about the axis is arbitrary, so
keys are computed from the axis-projection (completion-independent) and the
output carries the raw sub-resolution perpendicular dust -- poses agree up
to the dust size (< axial_rtol * scale), never O(scale).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Canon3DResult(NamedTuple):
    """points: (n,3) canonical pose, rows in canonical order.
    order: permutation of input row indices realizing that row order.
    regime: 'WWV' (the proposed PCA-free construction) or
        'R0' | 'R1' | 'R2' | 'R3' (spectrum regime, accelerated variant).
    n_candidates: size of the enumerated frame set."""

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


def _key_for_frame(X, R, scale, decimals):
    """Quantized total key + row order for the cloud expressed in frame R
    (columns of R = frame axes in ambient coordinates)."""
    Y = X @ R
    K = np.round(Y / scale * 10.0**decimals).astype(np.int64)
    order = np.lexsort(K.T[::-1])          # row-lexsort on quantized rows
    return tuple(K[order].ravel()), order, Y


def _complete_frame(z_hat, x_hat=None):
    """Right-handed frame (x, y, z) with the given z axis; if x_hat is None,
    complete with an arbitrary perpendicular (only used when the choice is
    provably immaterial: every point axial)."""
    if x_hat is None:
        a = np.array([1.0, 0.0, 0.0])
        if abs(z_hat @ a) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        x_hat = a - (a @ z_hat) * z_hat
        x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    return np.column_stack([x_hat, y_hat, z_hat])


def _axial_candidates(X, z_hat, scale, decimals, axial_rtol):
    """Candidate frames for one choice of axis direction z_hat: one proper +
    one improper frame per reference-ring point (WWV axial reduction: rings =
    equal quantized (rho, h); the reference ring is the lexicographically
    smallest such pair -- an equivariant choice).

    Points with rho < axial_rtol * scale are treated as AXIAL: normalizing a
    near-zero perpendicular component amplifies float error by scale/rho, so
    such points cannot define a numerically stable frame direction (they can
    still be represented -- only their use as candidates is excluded; the
    exclusion is by an invariant scalar, hence equivariant). Perpendicular
    structure below axial_rtol * scale is therefore sub-resolution by design.
    """
    q = 10.0**decimals
    h = X @ z_hat
    perp = X - np.outer(h, z_hat)
    rho = np.linalg.norm(perp, axis=1)
    rho_key = np.round(rho / scale * q).astype(np.int64)
    h_key = np.round(h / scale * q).astype(np.int64)

    nonaxial = np.flatnonzero(rho >= axial_rtol * scale)
    if nonaxial.size == 0:
        # Collinear-at-resolution along z. The completion of the frame is
        # NOT equivariant (fixed ambient vector), so the KEY must not see
        # the sub-resolution perpendicular dust: keys are computed from the
        # cloud PROJECTED onto the axis (coordinates exactly (0, 0, h) in
        # any completion -> completion-independent, pose-stable). The
        # returned coordinates keep the raw dust, so outputs of different
        # poses agree up to the actual dust size (< axial_rtol * scale),
        # never by O(scale). [Day-3 review: FATAL finding, fixed.]
        return [(_complete_frame(z_hat), np.outer(h, z_hat))]
    ring_keys = list(zip(rho_key[nonaxial].tolist(), h_key[nonaxial].tolist()))
    best = min(ring_keys)
    ring = nonaxial[[i for i, k in enumerate(ring_keys) if k == best]]

    frames = []
    for i in ring:
        x_hat = perp[i] / rho[i]
        F = _complete_frame(z_hat, x_hat)
        frames.append((F, None))
        frames.append((F * np.array([1.0, -1.0, 1.0]), None))  # mirror, det -1
    return frames


def canonicalize_3d_wwv(
    P,
    decimals: int = 9,
    axial_rtol: float = 1e-6,
    assume_centered: bool = False,
) -> Canon3DResult:
    """THE proposed canonicalization: the PCA-free 2D construction in 3D.

    2D -> 3D dictionary: centroid -> centroid; sort by angle about the
    origin -> choose (axis, azimuth); Booth least rotation of the quantized
    (radius, gap) token string -> argmin of the quantized row key over
    shell x ring candidate frames; the 2D mirror pass -> the mirrored frame
    paired with every candidate. No eigendecomposition anywhere, so nothing
    degenerates on symmetric clouds -- symmetry only makes MORE candidates
    tie, and ties are quantized symmetries (harmless).

    Candidate frames: reference shell = points with the smallest positive
    quantized radius (an invariant choice); each shell point's direction is
    a candidate +z axis; per axis, the WWV axial reduction supplies azimuth
    candidates (_axial_candidates). |frames| = 2 for a generic cloud,
    O(|shell| * |ring|) for symmetric ones.
    """
    X = _validate(P)
    n = X.shape[0]
    input_mag = float(np.abs(X).max())
    if not assume_centered:
        X = X - X.mean(axis=0)
    r = np.linalg.norm(X, axis=1)
    scale = float(r.max())
    if scale <= 64 * np.finfo(float).eps * input_mag:
        return Canon3DResult(np.zeros_like(X), np.arange(n), "R0", 0)

    q = 10.0**decimals
    r_key = np.round(r / scale * q).astype(np.int64)
    usable = np.flatnonzero(r >= axial_rtol * scale)   # conditioning rule
    shell_val = r_key[usable].min()
    shell = usable[r_key[usable] == shell_val]
    frames = []
    for i in shell:
        frames.extend(_axial_candidates(X, X[i] / r[i], scale, decimals,
                                        axial_rtol))

    best_key = best_order = best_Y = None
    for R, X_for_key in frames:
        key, order, _ = _key_for_frame(X if X_for_key is None else X_for_key,
                                       R, scale, decimals)
        if best_key is None or key < best_key:
            best_key, best_order, best_Y = key, order, X @ R
    return Canon3DResult(best_Y[best_order], best_order, "WWV", len(frames))


def canonicalize_3d(
    P,
    decimals: int = 9,
    gap_rtol: float = 1e-6,
    axial_rtol: float = 1e-6,
    assume_centered: bool = False,
) -> Canon3DResult:
    """Eigenframe-accelerated variant (spectrum-dichotomy dispatch). Same
    argmin principle and guarantees as canonicalize_3d_wwv; kept as the
    fast path. See module docstring."""
    X = _validate(P)
    n = X.shape[0]
    # Float-precision floor is set by the RAW input magnitude: centering
    # cancellation error ~ eps * |input|. No absolute clamp -- a tetrahedron
    # at radius 1e-15 about the origin is perfectly well-resolved relative
    # to its own scale and must NOT collapse to R0. [Day-3 review: MAJOR.]
    input_mag = float(np.abs(X).max())
    if not assume_centered:
        X = X - X.mean(axis=0)

    lam, V = np.linalg.eigh(X.T @ X)          # ascending eigenvalues
    top = float(lam[-1])
    noise2 = (64 * np.finfo(float).eps * input_mag) ** 2
    if top <= max(noise2, 0.0):               # R0: everything at the centroid
        return Canon3DResult(np.zeros_like(X), np.arange(n), "R0", 0)

    scale = float(np.linalg.norm(X, axis=1).max())
    gap12_small = (lam[1] - lam[0]) <= gap_rtol * top
    gap23_small = (lam[2] - lam[1]) <= gap_rtol * top

    if not gap12_small and not gap23_small:   # R1: 8 sign frames
        regime = "R1"
        frames = [
            (V * np.array(s), None)
            for s in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1))
        ]
    elif gap12_small and gap23_small:         # R3: axes from reference shell
        regime = "R3"
        q = 10.0**decimals
        r = np.linalg.norm(X, axis=1)
        r_key = np.round(r / scale * q).astype(np.int64)
        # same conditioning rule as _axial_candidates: a direction X[i]/r[i]
        # is only usable when r[i] is not a noise-scale length
        usable = np.flatnonzero(r >= axial_rtol * scale)
        shell_val = r_key[usable].min()
        shell = usable[r_key[usable] == shell_val]
        frames = []
        for i in shell:
            z_hat = X[i] / r[i]
            frames.extend(_axial_candidates(X, z_hat, scale, decimals,
                                            axial_rtol))
    else:                                     # R2: eigen-axis (simple eigval)
        regime = "R2"
        w = V[:, 0] if gap23_small else V[:, 2]   # eigenvector of the simple one
        frames = []
        for sign in (1.0, -1.0):
            frames.extend(_axial_candidates(X, sign * w, scale, decimals,
                                            axial_rtol))

    best_key = best_order = best_Y = None
    for R, X_for_key in frames:
        key, order, _ = _key_for_frame(X if X_for_key is None else X_for_key,
                                       R, scale, decimals)
        if best_key is None or key < best_key:
            best_key, best_order, best_Y = key, order, X @ R
    return Canon3DResult(best_Y[best_order], best_order, regime, len(frames))
