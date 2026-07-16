import numpy as np
import pytest
from scipy.stats import ortho_group

from src.sign_invariance import canonical_radial
from src.shapes import (
    cube,
    cuboctahedron,
    dodecahedron,
    icosahedron,
    octahedron,
    replicate,
    rotate_z,
    tetrahedron,
)


def random_orthogonal(rng):
    """Random O(3) matrix: rotation or reflection, each equally likely."""
    return ortho_group.rvs(dim=3, random_state=rng)


def assert_same_geometry(X, Y, atol=1e-6):
    """A canonical form must be a rigid transform + permutation of the input:
    same multiset of radii and of pairwise distances."""
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(X, axis=1)),
        np.sort(np.linalg.norm(Y, axis=1)),
        atol=atol,
    )
    dX = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    dY = np.linalg.norm(Y[:, None] - Y[None, :], axis=-1)
    np.testing.assert_allclose(np.sort(dX.ravel()), np.sort(dY.ravel()), atol=atol)


def assert_invariant(X, rng, n_transforms=20, atol=1e-6, permute=True):
    """The canonical form is unchanged by any O(3) transform and permutation."""
    result = canonical_radial(X)
    assert_same_geometry(X, result, atol=atol)
    for _ in range(n_transforms):
        Q = random_orthogonal(rng)
        Z = X[rng.permutation(len(X))] if permute else X
        np.testing.assert_allclose(result, canonical_radial(Z @ Q), atol=atol)
    return result


def test_radial_invariance_random(seed=0):
    rng = np.random.default_rng(seed)

    # Generic clouds: a unique farthest point fixes the pole outright.
    for _ in range(200):
        n = int(rng.integers(4, 20))
        X = rng.normal(size=(n, 3))
        assert_invariant(X, rng, n_transforms=8)


def test_radial_distinguishes_reflections(seed=1):
    rng = np.random.default_rng(seed)

    # A chiral (non-mirror-symmetric) cloud and its reflection must map to the
    # same canonical form, since both handednesses are enumerated.
    for _ in range(50):
        X = rng.normal(size=(int(rng.integers(4, 12)), 3))
        for M in (
            np.diag([1.0, 1.0, -1.0]),   # reflection through the xy plane
            np.diag([-1.0, -1.0, -1.0]),  # inversion
            random_orthogonal(rng),
        ):
            if np.linalg.det(M) > 0:
                M = M @ np.diag([1.0, 1.0, -1.0])  # force an improper map
            np.testing.assert_allclose(
                canonical_radial(X), canonical_radial(X @ M), atol=1e-6
            )


def test_radial_pole_ties_on_sphere(seed=2):
    rng = np.random.default_rng(seed)

    # Every point shares the maximum radius, so every point is tried as the
    # pole. The minimum over all of them must still be invariant.
    for _ in range(50):
        n = int(rng.integers(4, 12))
        dirs = rng.normal(size=(n, 3))
        X = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        assert_invariant(X, rng, n_transforms=10)


def test_radial_platonic_solids(seed=3):
    rng = np.random.default_rng(seed)

    # Maximum stress for the pole/projection symmetry handling: all vertices
    # tie in radius and every axis-aligned projection is highly symmetric, so
    # only the full-3d tie-break keeps the form well defined.
    for solid in (tetrahedron(), octahedron(), cube()):
        assert_invariant(solid, rng, n_transforms=20)


def test_radial_prisms_and_axial_symmetry(seed=4):
    rng = np.random.default_rng(seed)

    # Prisms: two parallel regular k-gons. The xy projection has k-fold
    # symmetry the 3d shape only shares if the two rings coincide in z, so z
    # must decide the frame.
    for k in (3, 4, 6):
        ang = 2 * np.pi * np.arange(k) / k
        ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(k)])
        prism = np.vstack([ring + [0, 0, 1.0], 0.6 * ring + [0, 0, -1.0]])
        assert_invariant(prism, rng, n_transforms=15)

    # Antiprism: the top ring is rotated, breaking the vertical mirror.
    ang = 2 * np.pi * np.arange(6) / 6
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(6)])
    top = np.column_stack(
        [np.cos(ang + np.pi / 6), np.sin(ang + np.pi / 6), np.zeros(6)]
    )
    antiprism = np.vstack([ring + [0, 0, 1.0], top + [0, 0, -1.0]])
    assert_invariant(antiprism, rng, n_transforms=15)


def test_radial_dense_asymmetric_shell(seed=13):
    rng = np.random.default_rng(seed)

    # Many points share the maximum radius, so every one is a pole candidate
    # and each projects the rest differently. With no symmetry exactly one pole
    # wins, and the argmin over all of them must land on the same physical
    # frame after any O(3) transform and permutation.
    for _ in range(30):
        n = int(rng.integers(8, 30))
        dirs = rng.normal(size=(n, 3))
        shell = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        assert_invariant(shell, rng, n_transforms=15)


def test_radial_shell_plus_asymmetric_interior(seed=14):
    rng = np.random.default_rng(seed)

    # A shell of equal-radius points (many pole candidates) plus a few strictly
    # interior points that break every symmetry: the projection to compare
    # differs for each pole, so the correct one must be chosen consistently.
    for _ in range(30):
        k = int(rng.integers(4, 12))
        dirs = rng.normal(size=(k, 3))
        shell = 2.0 * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        interior = 0.4 * rng.normal(size=(int(rng.integers(1, 5)), 3))
        X = np.vstack([shell, interior])
        assert_invariant(X, rng, n_transforms=15)


def test_radial_partial_max_radius_ties(seed=15):
    rng = np.random.default_rng(seed)

    # Exactly k points sit on the max-radius sphere and the rest are clearly
    # inside, so only the k are pole candidates. Sweeps the number of tied
    # poles from a couple up to many.
    for k in (2, 3, 5, 8):
        for _ in range(10):
            dirs = rng.normal(size=(k, 3))
            poles = 3.0 * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            inner = 0.3 * rng.normal(size=(6, 3))  # radius well below 3
            X = np.vstack([poles, inner])
            assert_invariant(X, rng, n_transforms=15)


def test_radial_icosahedron(seed=16):
    rng = np.random.default_rng(seed)

    # 12 vertices, all tied at the maximum radius: 12 pole candidates, each
    # with a symmetric projection, so the full-3d tie-break must agree across
    # every one of them.
    ico = icosahedron()
    assert ico.shape == (12, 3)
    assert_invariant(ico, rng, n_transforms=20)


def test_radial_subtol_max_radius_ties(seed=17):
    rng = np.random.default_rng(seed)

    # Radii jittered below tol must all still count as poles: a cloud that is
    # symmetric up to sub-tol radial noise must behave as exactly tied (any
    # pole gives the same canonical form up to that noise).
    for _ in range(20):
        jitter = 1e-12 * rng.normal(size=(6, 1))
        X = octahedron() * (1.0 + jitter)
        assert_invariant(X, rng, n_transforms=15)


def test_radial_dodecahedron(seed=18):
    rng = np.random.default_rng(seed)

    # The other icosahedral-symmetry solid: 20 vertices (a cube plus three
    # golden rectangles), all tied at radius sqrt(3), but with 3-fold vertex
    # figures instead of the icosahedron's 5-fold. Many poles, deep symmetry.
    dodeca = dodecahedron()
    assert dodeca.shape == (20, 3)
    assert_invariant(dodeca, rng, n_transforms=15)


def test_radial_cuboctahedron(seed=19):
    rng = np.random.default_rng(seed)

    # A vertex-transitive Archimedean solid (O_h): 12 vertices, all permutations
    # of (+-1, +-1, 0). Its vertices lie in three perpendicular squares, so many
    # axis-aligned projections are degenerate -- a strong test of the z-tiebreak.
    cuboct = cuboctahedron()
    assert cuboct.shape == (12, 3)
    assert_invariant(cuboct, rng, n_transforms=15)


def test_radial_chiral_rotational_symmetry(seed=20):
    rng = np.random.default_rng(seed)

    # Purely rotational (chiral) symmetry C_k about an axis: a random motif
    # replicated by k rotations has no mirror plane, so the two reflection
    # branches genuinely differ and step 8 must pick one consistently -- the 3D
    # analogue of the 2D pinwheel test.
    for k in (2, 3, 5):
        for _ in range(5):
            motif = rng.normal(size=(2, 3))
            X = replicate(motif, [rotate_z(2 * np.pi * j / k) for j in range(k)])
            assert_invariant(X, rng, n_transforms=15)


def test_radial_improper_rotation_S4(seed=21):
    rng = np.random.default_rng(seed)

    # S_4 symmetry: rotate 90 degrees about z then reflect through the xy plane.
    # Invariant under this improper rotation but not under the pure rotation or
    # a plain mirror, so it stresses the reflection handling differently from
    # any proper-rotation case.
    S4 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    powers = [np.linalg.matrix_power(S4, j) for j in range(4)]
    for _ in range(10):
        motif = rng.normal(size=(1, 3))
        X = replicate(motif, powers)
        assert_invariant(X, rng, n_transforms=15)


def test_radial_inversion_symmetry(seed=22):
    rng = np.random.default_rng(seed)

    # Inversion centre (C_i): every point is paired with its negative. Very
    # common in practice and a distinct symmetry element (improper, order 2).
    for _ in range(20):
        half = rng.normal(size=(int(rng.integers(2, 8)), 3))
        X = np.vstack([half, -half])
        assert_invariant(X, rng, n_transforms=15)


def test_radial_bipyramid_axis_poles(seed=23):
    rng = np.random.default_rng(seed)

    # Dihedral D_kh bipyramids: a regular k-gon ring plus two apexes on the
    # axis. The apexes are the farthest points, so here the poles lie ON the
    # symmetry axis (they project to the origin) while the symmetric ring is
    # what gets canonicalized -- the opposite regime from the Platonic tests.
    for k in (3, 4, 6):
        ang = 2 * np.pi * np.arange(k) / k
        ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(k)])
        bipyramid = np.vstack([ring, [[0.0, 0.0, 1.5], [0.0, 0.0, -1.5]]])
        assert_invariant(bipyramid, rng, n_transforms=15)


def test_radial_dual_compound_two_shells(seed=24):
    rng = np.random.default_rng(seed)

    # Cube-plus-octahedron dual compound (O_h) with the two solids on distinct
    # radial shells. Only the outer shell (the octahedron here) supplies poles,
    # so the pole set is a symmetric strict subset of the cloud.
    compound = np.vstack([cube(), 2.0 * octahedron()])
    assert_invariant(compound, rng, n_transforms=15)


def test_radial_points_on_the_axis(seed=5):
    rng = np.random.default_rng(seed)

    # Points collinear with the pole project onto the origin and carry no
    # azimuth; they must be ordered by z and not corrupt the frame.
    for _ in range(30):
        pole = rng.normal(size=3)
        pole *= 5.0 / np.linalg.norm(pole)  # make it the farthest point
        on_axis = np.outer(rng.uniform(-1, 1, size=3), pole)
        rest = rng.normal(size=(4, 3))
        X = np.vstack([pole, on_axis, rest])
        assert_invariant(X, rng, n_transforms=10)


def test_radial_collinear_cloud(seed=6):
    rng = np.random.default_rng(seed)

    # Every point lies on a single line through the origin: the pole is one
    # end and no point is off-axis, so the in-plane rotation is a free symmetry
    # and the code takes the axially-symmetric fallback. The result is fixed by
    # the z ordering alone and must still be invariant.
    for _ in range(20):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        X = np.outer(rng.uniform(-3, 3, size=int(rng.integers(2, 6))), u)
        assert_invariant(X, rng, n_transforms=10)


def test_radial_duplicates_and_near_symmetric(seed=7):
    rng = np.random.default_rng(seed)

    # Exact duplicates must not break tie handling.
    X = np.array(
        [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 1.0], [-1.0, 0.5, -2.0]]
    )
    assert_invariant(X, rng, n_transforms=20)

    # Sub-tol perturbation of a symmetric cloud behaves as exactly symmetric;
    # a clearly-resolved perturbation is handled deterministically.
    for delta in (1e-12, 1e-4):
        X = octahedron()
        X[0, 0] += delta
        assert_invariant(X, rng, n_transforms=15)


def test_radial_edge_cases(seed=8):
    rng = np.random.default_rng(seed)

    # Empty cloud is its own canonical form.
    assert canonical_radial(np.empty((0, 3))).shape == (0, 3)

    # A single point: canonical form is the point on the +z axis at its radius.
    result = canonical_radial(np.array([[0.0, 3.0, 4.0]]))
    np.testing.assert_allclose(np.linalg.norm(result), 5.0, atol=1e-9)
    for _ in range(10):
        Q = random_orthogonal(rng)
        np.testing.assert_allclose(
            result, canonical_radial(np.array([[0.0, 3.0, 4.0]]) @ Q), atol=1e-6
        )

    # A cloud entirely at the origin is its own canonical form.
    np.testing.assert_allclose(canonical_radial(np.zeros((4, 3))), 0.0)

    # Only 3d clouds are accepted.
    with pytest.raises(ValueError):
        canonical_radial(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        canonical_radial(np.zeros((5,)))


def test_radial_scale(seed=9):
    rng = np.random.default_rng(seed)

    # tol is absolute, so the cloud must stay well away from it in either
    # direction.
    for scale in (1e6, 1e-3):
        X = scale * rng.normal(size=(8, 3))
        result = canonical_radial(X)
        for _ in range(15):
            Q = random_orthogonal(rng)
            np.testing.assert_allclose(
                result, canonical_radial(X @ Q), atol=1e-9 * scale
            )


if __name__ == "__main__":
    test_radial_invariance_random()
    test_radial_distinguishes_reflections()
    test_radial_pole_ties_on_sphere()
    test_radial_platonic_solids()
    test_radial_prisms_and_axial_symmetry()
    test_radial_dense_asymmetric_shell()
    test_radial_shell_plus_asymmetric_interior()
    test_radial_partial_max_radius_ties()
    test_radial_icosahedron()
    test_radial_subtol_max_radius_ties()
    test_radial_dodecahedron()
    test_radial_cuboctahedron()
    test_radial_chiral_rotational_symmetry()
    test_radial_improper_rotation_S4()
    test_radial_inversion_symmetry()
    test_radial_bipyramid_axis_poles()
    test_radial_dual_compound_two_shells()
    test_radial_points_on_the_axis()
    test_radial_collinear_cloud()
    test_radial_duplicates_and_near_symmetric()
    test_radial_edge_cases()
    test_radial_scale()
    print("ok")
