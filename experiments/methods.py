"""
Uniform interface over the canonicalizers we compare, so the visualizer and the
robustness harness can run any of them by name.

Every method is a callable X:(n,3) -> Y:(n,3) returning the canonical cloud in
canonical row order. Inputs are centred first for *all* methods, so the
comparison isolates the rotation/reflection/permutation canonicalization rather
than translation handling (ASUN centres internally; canonical_radial assumes a
centred cloud).
"""

import os
import sys

import numpy as np

from src.sign_invariance import canonical_radial as _radial
from src.sign_invariance import canonical_radial_polar as _radial_polar
from src.sign_invariance import canonical_pca as _pca

# ASUN lives in results_reproduction/code as a bare (non-package) module.
_ASUN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results_reproduction", "code")
)
if _ASUN_DIR not in sys.path:
    sys.path.append(_ASUN_DIR)
from canonicalize_3d import canonicalize_3d as _asun  # noqa: E402


def _center(X):
    X = np.asarray(X, dtype=float)
    return X - X.mean(axis=0)


def radial(X):
    """Our max-radius-pole canonicalizer (src.sign_invariance.canonical_radial)."""
    return _radial(_center(X))


def asun(X):
    """ASUN eigenframe canonicalizer (results_reproduction.canonicalize_3d)."""
    return _asun(_center(X), assume_centered=True).points


def radial_polar(X):
    """Max-radius pole reduced to the reused 2d polar reference rule
    (src.sign_invariance.canonical_radial_polar)."""
    return _radial_polar(_center(X))


def pca(X):
    """Naive PCA canonicalizer (src.sign_invariance.canonical_pca): eigenframe
    with per-axis lexicographic sign, no degeneracy handling."""
    return _pca(_center(X))


METHODS = {"radial": radial, "radial_polar": radial_polar, "asun": asun, "pca": pca}
