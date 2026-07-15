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
