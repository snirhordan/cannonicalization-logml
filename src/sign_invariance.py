import numpy as np


def canonical_pca(X: np.ndarray, tol: float = 1e-12):
    """
    Canonicalizes X, a matrix that represents a 3d point cloud. We compute an
    invariant transformation to rotation, permutation and reflection.

    Parameters
    ----------
    X : array-like, shape (n_samples, 3)
        The input point cloud.
    tol : float, optional
        Tolerance for determining the sign of the eigenvectors. Default is 1e-12

    Returns
    -------
    Y : array-like, shape (n_samples, 3)
        The canonicalized point cloud.
    """
    X = np.asarray(X, dtype=float)

    # Add rotation invariance
    G = X.T @ X
    _, eigenvectors = np.linalg.eigh(G)
    Y = X @ eigenvectors

    # The eigenvectors are defined up to a sign, so we need to canonicalize the sign
    # The vectors may be permuted so to use a permutation invariant method, we sort
    # and compare lexicographic order.
    for i in range(3):
        v = Y[:, i]
        m_v = -v
        sorted_v = np.sort(v)
        sorted_m_v = np.sort(m_v)
        first_nonequal_index = np.argmax(sorted_v != sorted_m_v)
        if sorted_m_v[first_nonequal_index] > sorted_v[first_nonequal_index]:
            Y[:, i] = m_v

    # Add permutation invariance
    order = np.lexsort(Y.T[::-1])
    Y = Y[order]

    return Y


def _canonical_orientation(X: np.ndarray, tol: float):
    """
    Canonicalizes the rotation of a 2d point cloud for one fixed orientation
    (steps 1-2 and 4-7 of canonical_polar).

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        The input point cloud.
    tol : float
        Absolute tolerance for tie comparisons on radii and angles.

    Returns
    -------
    Y : array-like, shape (n_samples, 2)
        The rotated cloud, rows reordered canonically: points at the origin
        first, then the remaining points in cyclic angular order starting at
        the chosen reference point.
    key : tuple of two arrays
        The sequences of relative angles and radii starting at the chosen
        point, used to compare the two orientations in step 8.
    """
    r = np.linalg.norm(X, axis=1)
    nonzero = np.flatnonzero(r > tol)
    origin = np.flatnonzero(r <= tol)

    # Points at the origin are rotation invariant and have no well-defined
    # angle, so they are excluded from the angular analysis.
    if nonzero.size == 0:
        return X.copy(), (np.empty(0), np.empty(0))

    # Step 1: polar coordinates, sorted by angle. Points in the same direction
    # have no angular order that is stable under rotation (float noise decides
    # it), so points whose consecutive angles are within tol -- wrap-aware
    # around 0/2*pi -- form a group ordered by radius instead.
    alpha = np.mod(np.arctan2(X[nonzero, 1], X[nonzero, 0]), 2 * np.pi)
    r_nz = r[nonzero]
    idx = np.argsort(alpha, kind="stable")
    a_s = alpha[idx]
    m = idx.size

    # Step 2: relative angle to the next point (cyclic, sums to 2*pi).
    # Within a group the relative angle is snapped to 0, since sub-tol angle
    # differences are treated as equal everywhere else too.
    gaps = np.diff(a_s, append=a_s[0] + 2 * np.pi)
    breaks = np.flatnonzero(gaps > tol)
    if breaks.size == 0:
        # every consecutive gap is within tol: a single direction (or a cloud
        # denser than tol), so there is no angular structure at all
        final = idx[np.argsort(r_nz[idx], kind="stable")]
        rel = np.zeros(m)
        rel[-1] = 2 * np.pi
    else:
        # start the cyclic frame at a group boundary so groups are contiguous
        shift = (breaks[0] + 1) % m
        pos = np.roll(np.arange(m), -shift)
        gaps_r = gaps[pos]
        group = np.zeros(m, dtype=int)
        group[1:] = np.cumsum(gaps_r[:-1] > tol)
        # radius sorts within groups; the reorder stays inside each group, so
        # the group boundaries (and the gaps at them) keep their positions
        pos = pos[np.lexsort((r_nz[idx][pos], group))]
        rel = np.where(gaps_r > tol, gaps_r, 0.0)
        final = idx[pos]

    order = nonzero[final]
    a_sorted = alpha[final]
    r_sorted = r[order]

    # Step 4: candidates with minimum radius
    cand = np.flatnonzero(r_sorted <= r_sorted.min() + tol)

    # Step 5: among them, minimum relative angle
    if cand.size > 1:
        cand = cand[rel[cand] <= rel[cand].min() + tol]

    # Step 6: break remaining ties by walking to the following points and
    # keeping the candidates whose successor has lowest (radius, relative
    # angle). After one full circle, k survivors means a k-fold symmetry and
    # any of them yields the same canonical form.
    offset = 1
    while cand.size > 1 and offset < m:
        nxt = (cand + offset) % m
        keep = r_sorted[nxt] <= r_sorted[nxt].min() + tol
        cand, nxt = cand[keep], nxt[keep]
        if cand.size > 1:
            cand = cand[rel[nxt] <= rel[nxt].min() + tol]
        offset += 1
    i0 = cand[0]

    # Step 7: rotate so the chosen point lands on the positive y axis
    theta = np.pi / 2 - a_sorted[i0]
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Y = X @ R.T

    row_order = np.concatenate([origin, np.roll(order, -i0)])
    key = (np.roll(rel, -i0), np.roll(r_sorted, -i0))
    return Y[row_order], key


def canonical_polar(X: np.ndarray, tol: float = 1e-9):
    """
    Canonicalizes X, a matrix that represents a centered 2d point cloud. We compute an
    invariant transformation to rotation, reflection and permutation, by
    sorting the points by angle, choosing a canonical reference point from the
    polar coordinates (radius, relative angle to the next point) and rotating
    it onto the y axis. Both orientations of the cloud are canonicalized and
    the one with the lexicographically smallest sequence of relative angles
    (radii break exact ties) is returned.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        The input point cloud.
    tol : float, optional
        Absolute tolerance for tie comparisons on radii and angles. Default
        is 1e-9. The canonical form is stable when distinct radii/angles
        differ by much more than tol (or much less, in which case they are
        treated as equal); values differing by approximately tol sit on a
        knife edge, as with any tolerance-based canonicalization. Since the
        tolerance is absolute, the cloud should live on a scale well away
        from tol.

    Returns
    -------
    Y : array-like, shape (n_samples, 2)
        The canonicalized point cloud, rows in canonical cyclic order starting
        at the chosen reference point (points at the origin, if any, first).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"expected a point cloud of shape (n, 2), got {X.shape}")

    # Step 3: reflect across the x axis to invert the direction of travel
    Xp = X * np.array([1.0, -1.0])

    Y1, (rel1, r1) = _canonical_orientation(X, tol)
    Y2, (rel2, r2) = _canonical_orientation(Xp, tol)

    # Step 8: choose the orientation with the smallest sequence of relative
    # angles; if those coincide the radii sequences decide, and if those
    # coincide too the cloud is mirror symmetric and both agree.
    for seq1, seq2 in ((rel1, rel2), (r1, r2)):
        diff = seq1 - seq2
        significant = np.flatnonzero(np.abs(diff) > tol)
        if significant.size:
            return Y1 if diff[significant[0]] < 0 else Y2
    return Y1


def _axis_frame(u: np.ndarray) -> np.ndarray:
    """
    Orthonormal frame (returned as rows) whose third axis is the unit vector
    ``u``, so that ``R @ u == e_z``. The first two axes are arbitrary: the
    rotation about the pole that they leave free is fixed afterwards by the
    in-plane canonicalization, so any consistent choice works.
    """
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return np.stack([e1, e2, u])


def _lex_less(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """True if ``a`` precedes ``b`` in tol-tolerant lexicographic order."""
    diff = a - b
    significant = np.flatnonzero(np.abs(diff) > tol)
    return significant.size > 0 and diff[significant[0]] < 0


def canonical_radial(X: np.ndarray, tol: float = 1e-9):
    """
    Canonicalizes X, a matrix representing a (centered) 3d point cloud, to an
    invariant form under rotation, reflection and permutation.

    The 3d rotation group cannot be flattened to a single sortable coordinate
    the way SO(2) can, so we instead pick a pole from the data. The farthest
    point from the origin is a rotation/reflection *equivariant* choice: its
    radius is invariant, so it is the same physical point after any orthogonal
    transform, and its direction rotates with the cloud. Aligning that point to
    the +z axis removes two of the three rotational degrees of freedom; what
    remains is exactly O(2) acting on the xy plane (rotation about z plus
    in-plane reflection, both preserving z), which is the 2d problem solved by
    canonical_polar.

    The residual O(2) and reflection are canonicalized by enumeration, the
    direct 3d analogue of the 2d "rotate the reference point onto the y axis,
    try both orientations, keep the smaller" rule:

      * pole: every point tied for the maximum radius is tried;
      * in-plane rotation: every off-axis point is rotated onto the +y axis,
        the finite rotation-equivariant set of candidate frames;
      * reflection: both handednesses of the plane are tried.

    For each combination the cloud is sorted lexicographically in *full 3d* --
    so the z coordinate breaks ties that the xy projection cannot see -- and
    the lexicographically smallest sorted cloud is returned.

    Parameters
    ----------
    X : array-like, shape (n_samples, 3)
        The input point cloud. Assumed centered (radii are measured from the
        origin), matching canonical_polar.
    tol : float, optional
        Absolute tolerance for tie comparisons. Default is 1e-9. As with
        canonical_polar the cloud should live on a scale well away from tol.

    Returns
    -------
    Y : array-like, shape (n_samples, 3)
        The canonicalized point cloud.

    Notes
    -----
    The pole rides on a single extreme point, so unlike a PCA frame it is
    sensitive to noise/outliers near a radius tie; the payoff is that ties are
    discrete and enumerable rather than a continuous eigenspace. The
    enumeration costs O(k * m) candidate frames (k tied poles, m off-axis
    points), each an O(n log n) sort, so up to O(n^3 log n) overall -- fine for
    prototyping, not tuned for large clouds.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"expected a point cloud of shape (n, 3), got {X.shape}")
    if X.shape[0] == 0:
        return X.copy()

    r = np.linalg.norm(X, axis=1)
    rmax = r.max()
    if rmax <= tol:
        # The whole cloud sits at the origin: already O(3) invariant.
        return X[np.lexsort(X.T[::-1])].copy()

    poles = np.flatnonzero(r >= rmax - tol)

    best_rows = None
    best_key = None
    for p in poles:
        R = _axis_frame(X[p] / r[p])
        Xa = X @ R.T  # chosen pole -> +z, up to a free rotation about z

        xy0 = Xa[:, :2]
        z = Xa[:, 2]
        on_axis = np.hypot(xy0[:, 0], xy0[:, 1]) <= tol

        # Both handednesses of the plane: the 3d reflection that fixes the pole
        # reduces to a reflection of the projection (z is preserved either
        # way). Reflect first, then choose the reference, so the reference
        # always lands on +y -- otherwise the reflected coset is misaligned by
        # pi and a cloud and its mirror image would not agree.
        for sign in (1.0, -1.0):
            xy = xy0 * np.array([1.0, sign])
            xy[on_axis] = 0.0  # kill the float-noise azimuth of near-axis points

            # Candidate in-plane frames: bring each off-axis point's azimuth
            # onto the +y axis. This finite, rotation-equivariant set plays the
            # role of "rotate the reference point onto the y axis" from the 2d
            # algorithm. With no off-axis point the cloud is axially symmetric
            # and any orientation is canonical.
            refs = np.flatnonzero(~on_axis)
            angles = np.arctan2(xy[refs, 1], xy[refs, 0]) if refs.size else np.zeros(1)

            for a0 in angles:
                theta = np.pi / 2 - a0
                c, s = np.cos(theta), np.sin(theta)
                cloud = np.column_stack([xy @ np.array([[c, -s], [s, c]]).T, z])
                # Order rows (and compare candidates) on the coordinates
                # quantized to the tol grid. The reference point and the pole
                # both sit at x == 0, so a raw lexicographic sort would flip
                # their order on float noise; quantizing makes it stable while
                # still returning the real coordinates. Full 3d, so z breaks
                # ties the xy projection cannot see.
                quant = np.round(cloud / tol)
                order = np.lexsort((quant[:, 2], quant[:, 1], quant[:, 0]))
                key = quant[order].ravel()
                if best_key is None or _lex_less(key, best_key, 0.5):
                    best_rows, best_key = cloud[order], key

    return best_rows
