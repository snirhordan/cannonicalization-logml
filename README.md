# cannonicalization-logml

In this project we will implement algorithms for canonicalizing point clouds:
mapping a cloud to a canonical representative that is invariant under rotation,
reflection and permutation of the points, so that two clouds related by such a
transform map to the exact same array.

## Algorithms

All live in [`src/sign_invariance.py`](src/sign_invariance.py) and assume a
centered cloud (radii are measured from the origin).

### `canonical_polar` (2D)

Canonicalizes a 2D point cloud. SO(2) is one-dimensional, so the cloud can be
flattened to a single angular coordinate: sort the points by angle, pick a
canonical reference point from their polar coordinates (smallest radius, then
smallest relative angle to the next point, with a tie-breaking walk for
rotationally symmetric clouds), and rotate that point onto the +y axis. Both
reflections of the cloud are canonicalized and the one with the
lexicographically smaller (relative-angle, radius) sequence is returned.

### `canonical_radial` (3D)

Canonicalizes a 3D point cloud. SO(3) cannot be flattened to a single sortable
coordinate, so instead a pole is chosen from the data: the point farthest from
the origin. Its radius is invariant, so it is the same physical point after any
orthogonal transform, and aligning it to the +z axis removes two of the three
rotational degrees of freedom. What remains — rotation about that axis plus
reflection — is exactly the O(2) sub-problem that `canonical_polar` targets.

That residual is canonicalized by enumeration: every point tied for the maximum
radius is tried as the pole, every off-axis point is rotated onto the +y axis,
and both handednesses of the plane are tried; the candidate whose rows are
lexicographically smallest (sorted in full 3D) wins.

**Relation to `canonical_polar`.** `canonical_radial` reduces the 3D problem to
the same O(2) sub-problem, choosing the axis from the (invariant) radius. It is
*not* a literal call to `canonical_polar` on a 2D projection, for two reasons:

- The along-axis coordinate `z` is carried through as a tie-breaker in the
  ordering and selection. The bare projection is z-blind, so it would conflate
  points differing only in height and mis-handle projections that are more
  symmetric than the 3D shape.
- The pole is a choice that can tie many ways (a cloud on a sphere makes every
  point a candidate), and the canonical form minimizes over all tied poles.

The trade-off: the pole rides on a single extreme point, so it is more
sensitive to noise near a radius tie than an aggregate (e.g. PCA) frame would
be — but ties are discrete and enumerable rather than a continuous eigenspace.

### `canonical_pca` (3D)

An alternative 3D canonicalization that builds the frame from the covariance
eigenvectors (PCA) and fixes the per-axis sign. Stable for clouds with distinct
principal moments; degenerate (repeated) eigenvalues are its weak point.

## Tests

```
python -m pytest tests/
```
