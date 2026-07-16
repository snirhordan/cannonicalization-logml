import numpy as np


def _axis_frame(u):
    """Orthonormal frame whose third row is u, so X @ R.T sends u onto +z."""
    a = np.zeros(3)
    a[np.argmin(np.abs(u))] = 1.0
    e1 = np.cross(a, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return np.array([e1, e2, u])


def _lex_less(a, b, slack):
    """Lexicographic a < b, treating |a - b| <= slack as equal."""
    d = a - b
    i = np.flatnonzero(np.abs(d) > slack)
    return bool(i.size and d[i[0]] < 0)


def _inplane_reference(xy, z, lab, tol):
    """
    Steps 1-2 and 4-7 of canonical_polar on the xy projection, with (z, lab)
    appended to every tie comparison so that points sharing an xy position --
    and, crucially, rotationally symmetric *projections* of asymmetric clouds --
    are still ordered.

    Returns
    -------
    theta : float
        Rotation about z that puts the chosen reference point on the +y axis.
        Zero if the projection has no angular structure (all points on axis),
        in which case the cloud is axially symmetric and any theta is canonical.
    key : tuple of four arrays
        Cyclic (relative angle, radius, z, label) sequences starting at the
        reference, used by _canonical_inplane to pick the handedness.
    """
    r = np.linalg.norm(xy, axis=1)
    nonzero = np.flatnonzero(r > tol)

    # Points on the axis are fixed by every rotation about z *and* by the
    # reflection that fixes the pole, so they cannot inform either choice and
    # are excluded from the key entirely.
    if nonzero.size == 0:
        e = np.empty(0)
        return 0.0, (e, e, e, e)

    alpha = np.mod(np.arctan2(xy[nonzero, 1], xy[nonzero, 0]), 2 * np.pi)
    r_nz, z_nz, lab_nz = r[nonzero], z[nonzero], lab[nonzero]
    idx = np.argsort(alpha, kind="stable")
    a_s = alpha[idx]
    m = idx.size

    # Step 2: relative angle to the next point (cyclic, sums to 2*pi). Points
    # within tol in angle form a group with no rotation-stable angular order,
    # sorted by (radius, z, label) instead -- radius alone no longer suffices,
    # since distinct 3d points can share an xy position exactly.
    gaps = np.diff(a_s, append=a_s[0] + 2 * np.pi)
    breaks = np.flatnonzero(gaps > tol)
    if breaks.size == 0:
        final = idx[np.lexsort((lab_nz[idx], z_nz[idx], r_nz[idx]))]
        rel = np.zeros(m)
        rel[-1] = 2 * np.pi
    else:
        shift = (breaks[0] + 1) % m
        pos = np.roll(np.arange(m), -shift)
        gaps_r = gaps[pos]
        group = np.zeros(m, dtype=int)
        group[1:] = np.cumsum(gaps_r[:-1] > tol)
        rp, zp, lp = r_nz[idx][pos], z_nz[idx][pos], lab_nz[idx][pos]
        pos = pos[np.lexsort((lp, zp, rp, group))]
        rel = np.where(gaps_r > tol, gaps_r, 0.0)
        final = idx[pos]

    order = nonzero[final]
    a_sorted = alpha[final]
    r_sorted, z_sorted, lab_sorted = r[order], z[order], lab[order]

    # Steps 4-5: minimum radius, then minimum relative angle...
    cand = np.flatnonzero(r_sorted <= r_sorted.min() + tol)
    if cand.size > 1:
        cand = cand[rel[cand] <= rel[cand].min() + tol]
    # ...then z, then label. Without these two a cloud whose projection is
    # k-fold symmetric but whose z (or labels) are not would pick its reference
    # arbitrarily among the k, and the output would rotate with the input.
    if cand.size > 1:
        cand = cand[z_sorted[cand] <= z_sorted[cand].min() + tol]
    if cand.size > 1:
        cand = cand[lab_sorted[cand] == lab_sorted[cand].min()]

    # Step 6: walk to the following points, keeping the candidates whose
    # successor is lowest. k survivors after a full circle now means the
    # (rel, r, z, lab) sequence is k-periodic, i.e. the *3d* cloud has k-fold
    # symmetry about z, so any survivor yields the same canonical form.
    offset = 1
    while cand.size > 1 and offset < m:
        nxt = (cand + offset) % m
        for vals, slack in ((r_sorted, tol), (rel, tol), (z_sorted, tol),
                            (lab_sorted, 0.5)):
            if cand.size <= 1:
                break
            v = vals[nxt]
            keep = v <= v.min() + slack
            cand, nxt = cand[keep], nxt[keep]
        offset += 1
    i0 = cand[0]

    # Step 7: rotate the chosen point onto the positive y axis.
    theta = np.pi / 2 - a_sorted[i0]
    key = (np.roll(rel, -i0), np.roll(r_sorted, -i0),
           np.roll(z_sorted, -i0), np.roll(lab_sorted, -i0))
    return theta, key


def _canonical_inplane(xy0, z, lab, tol):
    """
    Canonicalize the O(2) that is left over once the pole sits on +z, by
    running the polar algorithm on the projection.

    Reflect first, then choose the reference, so the reference lands on +y in
    both branches -- otherwise the reflected coset is misaligned by pi.

    Returns (sign, theta): apply xy0 * [1, sign], then rotate by theta.
    """
    th1, k1 = _inplane_reference(xy0, z, lab, tol)
    th2, k2 = _inplane_reference(xy0 * np.array([1.0, -1.0]), z, lab, tol)

    # Step 8: relative angles decide, then radii, then z, then labels. Label
    # ranks are integers, so 0.5 is the right slack there whatever tol is.
    for s1, s2, slack in zip(k1, k2, (tol, tol, tol, 0.5)):
        diff = s1 - s2
        sig = np.flatnonzero(np.abs(diff) > slack)
        if sig.size:
            return (1.0, th1) if diff[sig[0]] < 0 else (-1.0, th2)
    # All four sequences agree: the cloud is mirror symmetric about a plane
    # through the pole and both branches produce the same 3d cloud.
    return 1.0, th1


def canonical_o3(X, labels=None, tol=1e-9):
    """
    Canonicalize a 3d point cloud under O(3) and permutation of the rows.

    The cloud is assumed to be centered already: this quotients out rotation
    and reflection about the origin, not translation.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"expected a point cloud of shape (n, 3), got {X.shape}")

    if labels is None:
        lab = np.zeros(X.shape[0], dtype=np.int64)
    else:
        labels = np.asarray(labels)
        if labels.shape != (X.shape[0],):
            raise ValueError(
                f"expected labels of shape ({X.shape[0]},), got {labels.shape}"
            )
        _, lab = np.unique(labels, return_inverse=True)
    lab = lab.astype(float)

    if X.shape[0] == 0:
        return (X, labels) if labels is not None else X

    r = np.linalg.norm(X, axis=1)
    rmax = r.max()
    if rmax <= tol:
        order = np.lexsort((lab, X[:, 2], X[:, 1], X[:, 0]))
        return (X[order], labels[order]) if labels is not None else X[order]

    poles = np.flatnonzero(r >= rmax - tol)
    poles = poles[lab[poles] == lab[poles].min()]

    best_cloud = best_order = best_key = None
    for p in poles:
        R = _axis_frame(X[p] / r[p])
        Xa = X @ R.T  # chosen pole -> +z, up to a free element of O(2)

        z = Xa[:, 2].copy()
        on_axis = np.hypot(Xa[:, 0], Xa[:, 1]) <= tol
        xy0 = np.where(on_axis[:, None], 0.0, Xa[:, :2])

        # One frame per pole, chosen by the polar algorithm rather than by
        # enumerating every off-axis point as a reference.
        sign, theta = _canonical_inplane(xy0, z, lab, tol)
        xy = xy0 * np.array([1.0, sign])
        c, s = np.cos(theta), np.sin(theta)
        cloud = np.column_stack([xy @ np.array([[c, -s], [s, c]]).T, z])

        # Score this single frame exactly as before: quantize to the tol grid
        # so rows sharing an x (the pole and the reference both sit at x == 0)
        # do not swap on float noise, sort, and compare poles on the result.
        quant = np.round(cloud / tol)
        order = np.lexsort((lab, quant[:, 2], quant[:, 1], quant[:, 0]))
        key = np.column_stack([quant, lab])[order].ravel()
        if best_key is None or _lex_less(key, best_key, 0.5):
            best_cloud, best_order, best_key = cloud, order, key

    Y = best_cloud[best_order]
    return (Y, labels[best_order]) if labels is not None else Y
