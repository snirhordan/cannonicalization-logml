"""Canonicalization of concentric 2D point clouds up to rotation (+ translation).

Implements the repaired version of the "minimal-gap" canonicalization, following
Wolter-Woo-Volz (1985) ORDER->ENCODE plus the Booth least-cyclic-rotation bridge
described in wiki/notes/wwv-symmetry-detection.md (Part 2, construction B),
restricted to a single ring in 2D.

Scope
-----
Domain: finite 2D clouds whose points are all equidistant from their centroid
(a single WWV "ring"). Group: SE(2) = rotations + translations by default;
pure SO(2) about the origin with ``assume_centered=True``. Reflections are NOT
canonicalized: mirror images generally receive different canonical poses.

Algorithm (fixed)
-----------------
1. Center at the centroid (translation invariance). All-at-origin -> return.
2. Validate the ring assumption: radii about the centroid equal within rtol.
   NOTE: "points on a circle" does not survive centering -- the centroid of a
   circle-inscribed cloud is generally not the circle's center. Equal radii
   must hold about the CENTROID; use assume_centered=True if your data is
   ring-shaped about a known origin instead.
3. Sort points clockwise starting from the +y axis; compute the cyclic string
   of angular gaps g_j (rotation-invariant, defined up to cyclic shift).
4. Encode each point as a token (quantized relative radius, quantized gap to
   the next point), quantized to ``decimals`` decimals as integers so exact
   ties survive float noise, and find the lexicographically least cyclic
   rotation with Booth's algorithm (O(k)). For an exact ring all radius keys
   are equal and the encoding reduces to the plain gap string; the radius
   component keeps the tie-break pose-independent even when the validator
   admits a small radius spread (a gap-only tie-break is radius-blind). The
   least rotation's start point p* is canonical: ties occur iff the token
   string is periodic, i.e. iff tied starts are related by an exact
   rotational symmetry of the cloud, so the output set is unaffected.
5. Rotate so p* lies on the +y ray. Read-out order from p* is the canonical
   ordering; the smallest period of the gap string gives the rotational order
   m of the cloud's symmetry group C_m for free (WWV CHECK step).

Why the naive rule fails
------------------------
The naive rule "pick the minimal gap; among ties take the maximal index" uses
indices assigned by the y-axis cut of the *input pose*, which rotation moves.
See canonicalize_2d_naive() and the counterexample in the test file. The lex-
least rotation subsumes it: the least rotation necessarily starts at a minimal
gap; it just breaks ties by the remaining string (pose-independent) instead of
by index (pose-dependent).

Exactness caveat
----------------
Exact invariance c(RP + t) = c(P) holds whenever quantization is faithful
(assumption (A_d)): (i) every exact gap lies farther than float noise from
every quantization boundary (m + 1/2) * 10^-decimals, so all poses quantize
each gap identically; and (ii) distinct cyclic rotations of the exact gap
string differ somewhere by more than the quantum, so quantized-equal readings
are exactly equal. When ties come from an eps-approximate symmetry (eps below
the quantum), tied starts are eps-approximately symmetry-related and the
output of different poses agrees up to O(k*eps*r) -- jitter. But when two
NON-symmetry-related readings tie only because gap values straddle a
quantization boundary, an arbitrarily small perturbation can flip the chosen
start and move the output by O(r): this boundary set is the (measure-zero,
unavoidable) discontinuity locus of the map (see
wiki/notes/impossibility-continuous-canonicalization.md). Rings whose radius
is below the float-precision floor of the input coordinates are collapsed to
the all-at-centroid case (angular structure is unresolvable there).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

TWO_PI = 2.0 * np.pi


class CanonResult(NamedTuple):
    """Result of canonicalize_2d.

    Attributes
    ----------
    points : (n, 2) canonical coordinates (canonical pose of the input).
    order : (n,) permutation of input indices, reading the ring clockwise
        from the canonical start point (canonical ordering by-product).
    rotational_order : detected rotational order m of the cloud (group C_m),
        from the smallest period of the quantized gap string. Sentinel 0 for
        the fully degenerate all-at-centroid cloud, whose stabilizer is the
        whole of SO(2), not a finite C_m.
    gap_string : (n,) canonical (lex-least) cyclic rotation of the quantized
        gap string, in radians.
    angle : CCW angle of the rotation that was applied to the centered cloud.
    """

    points: np.ndarray
    order: np.ndarray
    rotational_order: int
    gap_string: np.ndarray
    angle: float


def least_rotation_index(seq) -> int:
    """Booth's algorithm: index of the lexicographically least cyclic rotation.

    Returns the smallest index k such that seq[k:] + seq[:k] is lexicographically
    minimal among all cyclic rotations. O(len(seq)) time. Elements need only
    support ==, < comparisons (use integers for exact behaviour).
    """
    s = list(seq)
    n = len(s)
    if n == 0:
        raise ValueError("empty sequence")
    f = [-1] * (2 * n)  # failure function
    k = 0  # least rotation index found so far
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


def _smallest_period(key) -> int:
    """Smallest p > 0 with key[(i+p) % k] == key[i] for all i (p divides k).

    KMP search of key inside (key + key)[1:], O(k): the first match offset is
    the smallest cyclic shift fixing the string.
    """
    k = len(key)
    if k == 1:
        return 1
    fail = [0] * k
    j = 0
    for i in range(1, k):
        while j and key[i] != key[j]:
            j = fail[j - 1]
        if key[i] == key[j]:
            j += 1
            fail[i] = j
    text = key + key
    j = 0
    for i in range(1, 2 * k):  # guaranteed match at shift k (identity)
        while j and text[i] != key[j]:
            j = fail[j - 1]
        if text[i] == key[j]:
            j += 1
        if j == k:
            return i - k + 1
    return k


def _clockwise_angles_from_y(X: np.ndarray) -> np.ndarray:
    """Angle of each point measured CLOCKWISE from the +y axis, in [0, 2pi)."""
    phi = np.arctan2(X[:, 1], X[:, 0])  # CCW from +x
    return np.mod(np.pi / 2.0 - phi, TWO_PI)


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _center_and_validate(P, radius_rtol: float, assume_centered: bool):
    X = np.asarray(P, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] == 0:
        raise ValueError(f"expected a nonempty (n, 2) array, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("input contains non-finite coordinates")
    input_scale = float(np.abs(X).max())
    if not assume_centered:
        X = X - X.mean(axis=0)
    r = np.linalg.norm(X, axis=1)
    # Degenerate = ring radius at/below the float-precision floor of the raw
    # coordinates (centering cancellation noise ~ eps * input_scale): the
    # angular structure is unresolvable, so collapse to all-at-centroid.
    # The threshold is RELATIVE to the input scale, never absolute: a clean
    # radius-1e-7 ring about the origin must still canonicalize normally.
    noise_floor = 64.0 * np.finfo(float).eps * input_scale
    if r.max() <= noise_floor:
        return np.zeros_like(X), r, True
    scale = float(r.mean())
    if (r.max() - r.min()) > radius_rtol * scale:
        raise ValueError(
            "points are not equidistant from the centroid "
            f"(radii range [{r.min():.6g}, {r.max():.6g}]). Note that points on "
            "a common circle are generally NOT equidistant from their centroid; "
            "pass assume_centered=True if the ring center is the origin, or "
            "extend to multi-ring encoding for general clouds."
        )
    return X, r, False


def canonicalize_2d(
    P,
    decimals: int = 9,
    radius_rtol: float = 1e-6,
    assume_centered: bool = False,
) -> CanonResult:
    """Canonical pose + ordering of a concentric 2D cloud, up to rotation.

    Deterministic; invariant under rotations (and translations, unless
    assume_centered=True) whenever quantization at ``decimals`` is faithful
    (see module docstring). The output pose always has the canonical start
    point on the +y ray.
    """
    X, r, degenerate = _center_and_validate(P, radius_rtol, assume_centered)
    n = X.shape[0]
    if degenerate:
        # all points at the centroid: stabilizer is all of SO(2) (sentinel 0)
        return CanonResult(X, np.arange(n), 0, np.zeros(0), 0.0)

    theta = _clockwise_angles_from_y(X)
    sorted_idx = np.argsort(theta, kind="stable")  # clockwise from +y
    ts = theta[sorted_idx]
    gaps = np.mod(np.roll(ts, -1) - ts, TWO_PI)
    # k == 1: the single gap is the full circle. Reachable only with
    # assume_centered=True (a lone point is its own centroid, so the default
    # mode always takes the degenerate branch above).
    if n == 1:
        gaps = np.array([TWO_PI])
    gap_key = np.round(gaps * 10.0**decimals).astype(np.int64).tolist()
    # token = (relative radius of the j-th point, gap from it to the next);
    # radii are pose-invariant, so this stays a class function of cyclic
    # shifts while disambiguating starts a gap-only string cannot
    rad_key = (
        np.round(r[sorted_idx] / r.mean() * 10.0**decimals)
        .astype(np.int64)
        .tolist()
    )
    key = list(zip(rad_key, gap_key))

    b = least_rotation_index(key)
    order = sorted_idx[(b + np.arange(n)) % n]
    p_star = X[order[0]]
    angle = np.pi / 2.0 - np.arctan2(p_star[1], p_star[0])  # p* -> +y ray
    Xc = X @ _rotation(angle).T

    period = _smallest_period(key)
    # An all-zero gap string means the period shift is realized by a rotation
    # by 0 (coincident points): the true stabilizer is trivial C_1, not C_n.
    m = 1 if all(g == 0 for g in gap_key) else n // period
    canon_gap = np.array(gap_key[b:] + gap_key[:b], dtype=np.int64)
    return CanonResult(
        points=Xc,
        order=order,
        rotational_order=m,
        gap_string=canon_gap / 10.0**decimals,
        angle=float(np.mod(angle, TWO_PI)),
    )


def canonicalize_2d_naive(
    P,
    seed: Optional[int] = 0,
    radius_rtol: float = 1e-6,
    assume_centered: bool = False,
    tie_atol: float = 1e-9,
) -> np.ndarray:
    """The ORIGINAL (broken) proposal, kept verbatim for the counterexamples.

    1. center; 2. order points clockwise starting from the +y axis;
    3. find the minimal gap; 4. among points whose gap-to-next equals the
    minimal gap, pick the one with MAXIMAL INDEX (if all gaps are equal,
    pick uniformly at random); 5. rotate the chosen point onto the +y ray.

    Bug: the index in step 4 is assigned by the y-axis cut of the input pose,
    which is not rotation-invariant. See test_naive_counterexample.
    """
    X, r, degenerate = _center_and_validate(P, radius_rtol, assume_centered)
    n = X.shape[0]
    if degenerate:
        return X

    theta = _clockwise_angles_from_y(X)
    sorted_idx = np.argsort(theta, kind="stable")
    ts = theta[sorted_idx]
    gaps = np.mod(np.roll(ts, -1) - ts, TWO_PI)
    if n == 1:
        gaps = np.array([TWO_PI])

    candidates = np.flatnonzero(np.isclose(gaps, gaps.min(), atol=tie_atol))
    if len(candidates) == n and n > 1:  # "all points have the same angle"
        rng = np.random.default_rng(seed)
        choice = int(rng.integers(n))
    else:
        choice = int(candidates.max())  # "i is the maximal"  <- THE BUG
    p_star = X[sorted_idx[choice]]
    angle = np.pi / 2.0 - np.arctan2(p_star[1], p_star[0])
    return X @ _rotation(angle).T
