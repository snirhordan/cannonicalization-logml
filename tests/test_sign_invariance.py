import numpy as np
import pytest
from scipy.stats import ortho_group

from src.sign_invariance import _canonical_orientation, canonical_pca, canonical_polar


def random_orthogonal(rng):
    """Random O(3) matrix: rotation or reflection, each equally likely."""
    return ortho_group.rvs(dim=3, random_state=rng)


def rotation(theta):
    """2d rotation by theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def reflection(phi):
    """2d reflection about the axis through the origin at angle phi."""
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    return np.array([[c, s], [s, -c]])


def polar_cloud(degrees, radii):
    """Point cloud from polar coordinates, angles given in degrees."""
    a = np.deg2rad(degrees)
    return np.column_stack([radii * np.cos(a), radii * np.sin(a)])


def is_nondegenerate(eigenvalues, tol=1e-6):
    diffs = np.diff(np.sort(eigenvalues))
    return np.all(diffs > tol)


def test_invariance_under_permutation_and_orthogonal_transform(seed=0):
    rng = np.random.default_rng(seed)

    n_trials = 200
    n_checked = 0

    for _ in range(n_trials):
        n = rng.integers(4, 20)
        X = rng.normal(size=(n, 3))

        result = canonical_pca(X)

        if not is_nondegenerate(np.linalg.eigvalsh(X.T @ X)):
            continue

        n_checked += 1

        # Permute rows (points) -- Gram matrix over columns is unaffected.
        perm = rng.permutation(n)
        X_perm = X[perm]
        result_perm = canonical_pca(X_perm)

        np.testing.assert_allclose(result, result_perm, atol=1e-6)

        # Apply a random rotation/reflection: X' = X @ Q.
        Q = random_orthogonal(rng)
        X_rot = X @ Q
        result_rot = canonical_pca(X_rot)

        np.testing.assert_allclose(result, result_rot, atol=1e-6)

        # Combine permutation and rotation/reflection together.
        X_both = X_perm @ Q
        result_both = canonical_pca(X_both)

        np.testing.assert_allclose(result, result_both, atol=1e-6)

    print(f"Checked {n_checked} non-degenerate trials out of {n_trials} total trials.")

    assert n_checked > 0, "no non-degenerate trials were generated"


def test_polar_invariance_under_rotation_and_reflection(seed=0):
    rng = np.random.default_rng(seed)

    n_trials = 200
    n_rotations = 0
    n_reflections = 0

    for _ in range(n_trials):
        n = int(rng.integers(1, 20))
        X = rng.normal(size=(n, 2))

        result = canonical_polar(X)

        # The canonical form must preserve the geometry of the cloud: same
        # multiset of radii and of pairwise distances.
        np.testing.assert_allclose(
            np.sort(np.linalg.norm(X, axis=1)),
            np.sort(np.linalg.norm(result, axis=1)),
            atol=1e-6,
        )
        dists = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
        dists_result = np.linalg.norm(result[:, None] - result[None, :], axis=-1)
        np.testing.assert_allclose(
            np.sort(dists.ravel()), np.sort(dists_result.ravel()), atol=1e-6
        )

        # The chosen reference point (first row) lies on the positive y axis.
        assert abs(result[0, 0]) < 1e-6
        assert result[0, 1] > 0

        # Apply a random rotation/reflection: X' = X @ Q.
        Q = ortho_group.rvs(dim=2, random_state=rng)
        if np.linalg.det(Q) > 0:
            n_rotations += 1
        else:
            n_reflections += 1
        result_rot = canonical_polar(X @ Q)

        np.testing.assert_allclose(result, result_rot, atol=1e-6)

        # Row order is canonical too, so permutations don't change the result.
        perm = rng.permutation(n)
        result_both = canonical_polar(X[perm] @ Q)

        np.testing.assert_allclose(result, result_both, atol=1e-6)

    # ortho_group samples both components of O(2); make sure the trials
    # actually exercised rotations as well as reflections.
    assert n_rotations > 0, "no pure rotations were sampled"
    assert n_reflections > 0, "no reflections were sampled"


def test_polar_invariance_under_reflection_about_any_axis(seed=1):
    rng = np.random.default_rng(seed)

    for _ in range(100):
        n = int(rng.integers(1, 20))
        X = rng.normal(size=(n, 2))
        result = canonical_polar(X)

        # Pure reflection about an arbitrary axis through the origin.
        M = reflection(rng.uniform(0, 2 * np.pi))
        assert np.isclose(np.linalg.det(M), -1.0)

        np.testing.assert_allclose(result, canonical_polar(X @ M), atol=1e-6)

        # The same reflection composed with a rotation.
        MR = M @ rotation(rng.uniform(0, 2 * np.pi))
        np.testing.assert_allclose(result, canonical_polar(X @ MR), atol=1e-6)


def test_polar_invariance_on_symmetric_clouds(seed=2):
    rng = np.random.default_rng(seed)

    # Regular k-gons (k-fold symmetry) and nested k-gons with distinct radii:
    # every tie-breaking candidate must yield the same canonical form.
    for k in (3, 4, 6):
        angles = 2 * np.pi * np.arange(k) / k
        polygon = np.column_stack([np.cos(angles), np.sin(angles)])
        nested = np.vstack([polygon, 2.5 * polygon @ rotation(np.pi / k)])

        for X in (polygon, nested):
            result = canonical_polar(X)
            for _ in range(20):
                Q = ortho_group.rvs(dim=2, random_state=rng)
                np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)


def test_polar_tie_breaking(seed=3):
    rng = np.random.default_rng(seed)

    # Each cloud is built so that a specific tie-breaking step decides, and so
    # that the expected winner is the point at angle 0: the canonicalized
    # orientation must report exactly its (relative angle, radius) sequence.
    cases = [
        # step 5: two points share the minimum radius (angles 0 and 100),
        # the relative angle decides (30 vs 100 degrees)
        ([0, 30, 100, 200], [1, 2, 1, 3]),
        # step 6: the relative angles tie too (both 40 degrees), the radius
        # of the first successor decides (2 vs 3)
        ([0, 40, 180, 220], [1, 2, 1, 3]),
        # step 6, second iteration: the first successors tie in radius and
        # relative angle as well, only the second successor decides (3 vs 4)
        ([0, 40, 80, 180, 220, 260], [1, 2, 3, 1, 2, 4]),
    ]

    for degrees, radii in cases:
        X = polar_cloud(degrees, radii)
        rel_expected = np.deg2rad(np.diff(degrees, append=degrees[0] + 360))

        _, (rel_seq, r_seq) = _canonical_orientation(X, tol=1e-9)
        np.testing.assert_allclose(r_seq, radii, atol=1e-9)
        np.testing.assert_allclose(rel_seq, rel_expected, atol=1e-9)

        # The selection must pick the same physical point after any rotation.
        for _ in range(10):
            X_rot = X @ rotation(rng.uniform(0, 2 * np.pi)).T
            _, (rel_rot, r_rot) = _canonical_orientation(X_rot, tol=1e-9)
            np.testing.assert_allclose(r_rot, radii, atol=1e-6)
            np.testing.assert_allclose(rel_rot, rel_expected, atol=1e-6)

        # And the full pipeline stays invariant under all of O(2).
        result = canonical_polar(X)
        for _ in range(20):
            Q = ortho_group.rvs(dim=2, random_state=rng)
            np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)


def test_polar_distinguishes_reflections():
    # A scalene cloud is not mirror symmetric, yet X and its reflection must
    # map to the same canonical form.
    X = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, -1.0], [0.5, 0.5]])
    X_reflected = X * np.array([1.0, -1.0])

    result = canonical_polar(X)
    result_reflected = canonical_polar(X_reflected)

    np.testing.assert_allclose(result, result_reflected, atol=1e-6)


def test_polar_edge_cases(seed=4):
    rng = np.random.default_rng(seed)

    # Points at the origin have no angle: they are kept, placed first, and do
    # not break the invariance of the rest of the cloud.
    X = np.vstack([np.zeros((2, 2)), rng.normal(size=(5, 2))])
    result = canonical_polar(X)
    np.testing.assert_allclose(result[:2], 0.0, atol=1e-9)
    assert abs(result[2, 0]) < 1e-6
    assert result[2, 1] > 0
    for _ in range(20):
        Q = ortho_group.rvs(dim=2, random_state=rng)
        perm = rng.permutation(len(X))
        np.testing.assert_allclose(result, canonical_polar(X[perm] @ Q), atol=1e-6)

    # A cloud that is entirely at the origin is its own canonical form.
    np.testing.assert_allclose(canonical_polar(np.zeros((3, 2))), 0.0)

    # Duplicated points share both radius and angle and must not break the
    # tie handling.
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-1.5, 0.5]])
    result = canonical_polar(X)
    for _ in range(20):
        Q = ortho_group.rvs(dim=2, random_state=rng)
        np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)

    # Mirror-symmetric cloud: the two orientations tie all the way through
    # step 8 and must agree.
    X = np.array([[0.0, 1.0], [1.0, 2.0], [-1.0, 2.0]])
    result = canonical_polar(X)
    for _ in range(20):
        Q = ortho_group.rvs(dim=2, random_state=rng)
        np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)

    # Only 2d clouds are accepted.
    with pytest.raises(ValueError):
        canonical_polar(np.zeros((5, 3)))


def test_polar_collinear_same_direction(seed=5):
    rng = np.random.default_rng(seed)

    # Points lying in exactly the same direction have no angular order: after
    # a rotation, float noise (~1e-16 rad) decides their sort order unless
    # the radius does. This broke the pre-grouping implementation.
    clouds = [
        np.array([[1.0, 0.0], [2.0, 0.0], [0.5, 1.0]]),
        np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [-1.0, 0.5], [-2.0, 1.0]]),
    ]
    for X in clouds:
        result = canonical_polar(X)
        for _ in range(100):
            Q = ortho_group.rvs(dim=2, random_state=rng)
            perm = rng.permutation(len(X))
            np.testing.assert_allclose(result, canonical_polar(X[perm] @ Q), atol=1e-6)


def test_polar_direction_group_straddling_wrap(seed=6):
    rng = np.random.default_rng(seed)

    # Two points in nearly the same direction, one just below angle 0 and one
    # just above: the group must be detected across the 0/2*pi wrap.
    eps = 1e-13
    X = np.array(
        [
            [3 * np.cos(-eps), 3 * np.sin(-eps)],
            [np.cos(eps), np.sin(eps)],
            [0.0, 2.0],
        ]
    )
    result = canonical_polar(X)
    for _ in range(100):
        Q = ortho_group.rvs(dim=2, random_state=rng)
        np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)


def test_polar_step6_picks_the_right_candidate(seed=7):
    rng = np.random.default_rng(seed)

    # Hexagon with radii [1,5,9,1,5,8]: two candidates tie through steps 4-5
    # and the first successor, and the walk resolves at the second successor
    # in favor of the *second* candidate (its chain reads 1,5,8,... which
    # beats 1,5,9,...). An implementation that just keeps the first candidate
    # passes every other test but fails here.
    X = polar_cloud([0, 60, 120, 180, 240, 300], [1, 5, 9, 1, 5, 8])

    _, (rel_seq, r_seq) = _canonical_orientation(X, tol=1e-9)
    np.testing.assert_allclose(r_seq, [1, 5, 8, 1, 5, 9], atol=1e-9)
    np.testing.assert_allclose(rel_seq, np.deg2rad(np.full(6, 60.0)), atol=1e-9)

    result = canonical_polar(X)
    for _ in range(30):
        Q = ortho_group.rvs(dim=2, random_state=rng)
        np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)


def test_polar_chiral_rotationally_symmetric_clouds(seed=8):
    rng = np.random.default_rng(seed)

    # Pinwheels: k-fold rotational symmetry but no mirror symmetry, so steps
    # 4-6 run a full circle of ties (k candidates survive) while the two
    # orientations of step 8 genuinely differ and must be chosen consistently.
    for k in (2, 3, 5):
        degrees = np.concatenate([np.array([0.0, 25.0]) + 360.0 * j / k for j in range(k)])
        radii = np.tile([1.0, 2.0], k)
        X = polar_cloud(degrees, radii)

        result = canonical_polar(X)
        for _ in range(30):
            Q = ortho_group.rvs(dim=2, random_state=rng)
            np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)


def test_polar_mirror_symmetric_about_random_axes(seed=9):
    rng = np.random.default_rng(seed)

    # Clouds built to be mirror symmetric about a random axis: both step-8
    # orientations describe the same cloud and must produce the same output.
    for _ in range(20):
        half = rng.normal(size=(int(rng.integers(1, 8)), 2))
        phi = rng.uniform(0, 2 * np.pi)
        X = np.vstack([half, half @ reflection(phi)])

        result = canonical_polar(X)
        for _ in range(10):
            Q = ortho_group.rvs(dim=2, random_state=rng)
            np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)


def test_polar_near_symmetric_perturbations(seed=10):
    rng = np.random.default_rng(seed)

    # Regular hexagon with one radius perturbed. Well below tol the cloud
    # must behave as exactly symmetric (any candidate is fine because the
    # outputs agree up to the perturbation); well above tol the perturbation
    # must be resolved deterministically. Perturbations of about tol are a
    # knife edge by design and are not tested.
    angles = np.arange(6) * 60.0
    for delta in (1e-12, -1e-12, 1e-5, -1e-5):
        radii = np.ones(6)
        radii[0] += delta
        X = polar_cloud(angles, radii)

        result = canonical_polar(X)
        for _ in range(30):
            Q = ortho_group.rvs(dim=2, random_state=rng)
            np.testing.assert_allclose(result, canonical_polar(X @ Q), atol=1e-6)


def test_polar_torture_cloud(seed=11):
    rng = np.random.default_rng(seed)

    # Everything at once: repeated origin points, exact duplicates, three
    # collinear points in the same direction, and generic points.
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, 3.0],
            [0.5, -2.0],
        ]
    )
    result = canonical_polar(X)
    for _ in range(100):
        Q = ortho_group.rvs(dim=2, random_state=rng)
        perm = rng.permutation(len(X))
        np.testing.assert_allclose(result, canonical_polar(X[perm] @ Q), atol=1e-6)


def test_polar_scale_and_empty(seed=12):
    rng = np.random.default_rng(seed)

    # An empty cloud is its own canonical form.
    assert canonical_polar(np.empty((0, 2))).shape == (0, 2)

    # Large and small coordinate scales (tol is absolute, so the cloud scale
    # must stay well away from tol).
    for scale in (1e6, 1e-6):
        X = scale * rng.normal(size=(10, 2))
        result = canonical_polar(X)
        for _ in range(20):
            Q = ortho_group.rvs(dim=2, random_state=rng)
            np.testing.assert_allclose(
                result, canonical_polar(X @ Q), atol=1e-9 * scale
            )


if __name__ == "__main__":
    test_invariance_under_permutation_and_orthogonal_transform()
    test_polar_invariance_under_rotation_and_reflection()
    test_polar_invariance_under_reflection_about_any_axis()
    test_polar_invariance_on_symmetric_clouds()
    test_polar_tie_breaking()
    test_polar_distinguishes_reflections()
    test_polar_edge_cases()
    test_polar_collinear_same_direction()
    test_polar_direction_group_straddling_wrap()
    test_polar_step6_picks_the_right_candidate()
    test_polar_chiral_rotationally_symmetric_clouds()
    test_polar_mirror_symmetric_about_random_axes()
    test_polar_near_symmetric_perturbations()
    test_polar_torture_cloud()
    test_polar_scale_and_empty()
    print("ok")
