"""
Shared point-cloud constructions used by the tests, the visualizer, and the
robustness experiments. Keeping them in one place avoids the same solids
drifting apart across three files.

The fixed named solids (`tetrahedron`, ..., `cuboctahedron`) return canonical
vertex sets; `build_cases` returns a representative dict of clouds spanning the
symmetry classes we care about.
"""

import numpy as np

PHI = (1 + 5**0.5) / 2  # golden ratio


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def rotate_z(theta):
    """Rotation by ``theta`` about the z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def replicate(motif, matrices):
    """Union of ``motif`` under each transform in ``matrices`` (rows = points)."""
    return np.vstack([motif @ M.T for M in matrices])


def signs(*vals):
    """All sign combinations, e.g. ``signs(-1, 1)`` -> the 8 cube corners."""
    return [[a, b, c] for a in vals for b in vals for c in vals]


# --------------------------------------------------------------------------- #
# Fixed solids
# --------------------------------------------------------------------------- #
def tetrahedron():
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


def octahedron():
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )


def cube():
    return np.array(signs(-1.0, 1.0))


def icosahedron():
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts += [[0, s1, s2 * PHI], [s1, s2 * PHI, 0], [s1 * PHI, 0, s2]]
    return np.array(verts)


def dodecahedron():
    verts = list(signs(-1.0, 1.0))
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts += [
                [0, s1 / PHI, s2 * PHI],
                [s1 / PHI, s2 * PHI, 0],
                [s1 * PHI, 0, s2 / PHI],
            ]
    return np.array(verts)


def cuboctahedron():
    verts = []
    for a, b in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        verts += [[a, b, 0], [a, 0, b], [0, a, b]]
    return np.array(verts)


# --------------------------------------------------------------------------- #
# Representative case set
# --------------------------------------------------------------------------- #
def build_cases(seed=0):
    """A representative dict {name: (n, 3) cloud} spanning symmetry classes."""
    rng = np.random.default_rng(seed)
    cases = {}

    cases["generic"] = rng.normal(size=(9, 3))

    cases["tetrahedron"] = tetrahedron()
    cases["octahedron"] = octahedron()
    cases["cube"] = cube()
    cases["icosahedron"] = icosahedron()
    cases["dodecahedron"] = dodecahedron()
    cases["cuboctahedron"] = cuboctahedron()

    ang = 2 * np.pi * np.arange(6) / 6
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros(6)])
    cases["prism"] = np.vstack([ring + [0, 0, 1.0], 0.6 * ring + [0, 0, -1.0]])
    top = np.column_stack(
        [np.cos(ang + np.pi / 6), np.sin(ang + np.pi / 6), np.zeros(6)]
    )
    cases["antiprism"] = np.vstack([ring + [0, 0, 1.0], top + [0, 0, -1.0]])
    cases["bipyramid"] = np.vstack([ring, [[0, 0, 1.5], [0, 0, -1.5]]])

    cases["chiral_C3"] = replicate(
        rng.normal(size=(2, 3)), [rotate_z(2 * np.pi * j / 3) for j in range(3)]
    )
    S4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    cases["improper_S4"] = replicate(
        rng.normal(size=(1, 3)), [np.linalg.matrix_power(S4, j) for j in range(4)]
    )
    half = rng.normal(size=(4, 3))
    cases["inversion"] = np.vstack([half, -half])

    cases["dual_compound"] = np.vstack([cube(), 2.0 * octahedron()])

    dirs = rng.normal(size=(12, 3))
    cases["sphere_shell"] = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    return cases
