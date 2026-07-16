"""methods.py -- uniform canonicalization wrappers for the ASUN-robustness harness.

Four methods, each exposed as a function (pos, species) -> MethodResult:

  asun      ASUN (Baker et al. 2024, ICML), CategoricalPointCloud.CatFrame,
            tol=1e-2 (their shipped default, never tuned). Gets species --
            its strongest, most symmetry-reducing form.
  pca       Sign-fixed PCA: eigenframe of the centered Gram X^T X, with a
            deterministic third-moment (skewness) sign rule per axis to
            resolve the (Z/2)^3 reflection ambiguity, and a determinant fix
            for the O(3)/SO(3) handedness ambiguity. Species-blind.
  ours      canonicalize_3d (eigenframe-accelerated regime-dispatch argmin).
            Species-blind (deliberately -- see README "fairness").
  ours_wwv  canonicalize_3d_wwv (the PCA-free proposal). Species-blind.

Every wrapper is wrapped in try/except and NEVER lets an exception escape:
on failure it returns crashed=True with the exception type name recorded,
and pose=None (a crash is a crash -- it must never be silently scored as a
perfect match or dropped from crash-rate accounting; see metrics.py).

ASUN is vendored under vendor/pyorbit/ (copied from
https://github.com/Utah-Math-Data-Science/alignment, commit 17d2c7f,
2024-07-25) so the harness does not depend on the orchestrator's ephemeral
job tmp directory. Ours is imported directly from
/home/snirhordan/linkedin/docs/logml/code/canonicalize_3d.py (the live
project code, not vendored, since it is our own, actively-developed module).
"""
from __future__ import annotations

import os
import sys
from typing import NamedTuple, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.normpath(os.path.join(_HERE, "..", "..", "code"))
sys.path.insert(0, os.path.join(_HERE, "vendor", "pyorbit"))
sys.path.insert(0, _CODE)

from CategoricalPointCloud import CatFrame  # noqa: E402  (ASUN, species-aware)
from canonicalize_3d import canonicalize_3d, canonicalize_3d_wwv  # noqa: E402

ASUN_TOL = 1e-2  # ASUN's shipped default -- fixed, per task spec, never tuned.


class MethodResult(NamedTuple):
    pose: Optional[np.ndarray]          # (n,3) canonical pose, or None if crashed
    species_out: Optional[np.ndarray]   # (n,) species aligned row-for-row with pose
    crashed: bool
    err_type: Optional[str]             # exception class name, or None
    regime: Optional[str] = None        # 'R0'..'R3'/'WWV' for ours, else None
    n_candidates: Optional[int] = None  # candidate-frame count for ours, else None


def run_asun(pos: np.ndarray, species: np.ndarray) -> MethodResult:
    """ASUN CatFrame(tol=1e-2).get_frame(pos, species). Rows of the returned
    pose are NOT permuted relative to the input (align() only rotates), so
    species_out == species."""
    pos = np.asarray(pos, dtype=float)
    species = np.asarray(species)
    try:
        f = CatFrame(tol=ASUN_TOL)
        aligned, _frame = f.get_frame(pos.copy(), species.copy())
        aligned = np.asarray(aligned, dtype=float)
        if aligned.shape != pos.shape or not np.all(np.isfinite(aligned)):
            return MethodResult(None, None, True, "InvalidOutput")
        return MethodResult(aligned, species.copy(), False, None)
    except AssertionError:
        return MethodResult(None, None, True, "AssertionError")
    except Exception as e:  # noqa: BLE001 -- must catch everything, log the type
        return MethodResult(None, None, True, type(e).__name__)


def _sign_fixed_pca(X: np.ndarray) -> np.ndarray:
    """Eigenframe of the centered Gram C = X^T X.

    Axes ordered by DESCENDING eigenvalue (a deterministic function of the
    spectrum). Each axis's sign is fixed by the Bro et al. (2008) third-moment
    rule: sign so that sum_i (X @ v_i)_i^3 >= 0. This is invariant under
    global rotation of X (the projections transform equivariantly) so at
    sigma=0 the result is rotation-consistent PROVIDED the eigenvalues are
    non-degenerate (eigh's basis choice within a degenerate eigenspace is not
    itself equivariant -- this is the PCA-degeneracy dichotomy the harness is
    built to expose, so we do NOT paper over it with an extra tie-break rule
    for the degenerate case; it is a real, reported failure mode of the PCA
    baseline, not a bug in this wrapper).

    Tie-break when the third moment is numerically zero (symmetric
    distribution along that axis, e.g. any point cloud with an odd-order
    axis of symmetry): use the sign of the projection with the largest
    absolute value; if that is also ~0 (rare degenerate case), default +1.
    Handedness (det=+1) is fixed by flipping the LAST (smallest-variance)
    axis if needed, so the two remaining ambiguities are collapsed to one
    deterministic frame.
    """
    C = X.T @ X
    lam, V = np.linalg.eigh(C)          # ascending eigenvalues
    order = np.argsort(-lam)            # descending
    V = V[:, order]
    for i in range(3):
        v = V[:, i]
        proj = X @ v
        s3 = float(np.sum(proj ** 3))
        scale3 = float(np.sum(proj ** 2)) ** 1.5 + 1e-300
        if abs(s3) > 1e-9 * scale3:
            sign = 1.0 if s3 >= 0 else -1.0
        else:
            idx = int(np.argmax(np.abs(proj)))
            val = proj[idx]
            sign = 1.0 if val >= -1e-12 else -1.0
        V[:, i] = V[:, i] * sign
    if np.linalg.det(V) < 0:
        V[:, -1] = -V[:, -1]
    return X @ V


def run_pca(pos: np.ndarray, species: np.ndarray) -> MethodResult:
    pos = np.asarray(pos, dtype=float)
    species = np.asarray(species)
    try:
        X = pos - pos.mean(axis=0)
        pose = _sign_fixed_pca(X)
        if not np.all(np.isfinite(pose)):
            return MethodResult(None, None, True, "InvalidOutput")
        return MethodResult(pose, species.copy(), False, None)
    except Exception as e:  # noqa: BLE001
        return MethodResult(None, None, True, type(e).__name__)


def run_ours(pos: np.ndarray, species: np.ndarray) -> MethodResult:
    """canonicalize_3d -- geometry only, species is carried through by
    reindexing with the returned row order (ours reorders rows; ASUN/PCA do
    not)."""
    pos = np.asarray(pos, dtype=float)
    species = np.asarray(species)
    try:
        res = canonicalize_3d(pos)
        species_out = species[res.order]
        if not np.all(np.isfinite(res.points)):
            return MethodResult(None, None, True, "InvalidOutput")
        return MethodResult(res.points, species_out, False, None, res.regime, res.n_candidates)
    except Exception as e:  # noqa: BLE001
        return MethodResult(None, None, True, type(e).__name__)


def run_ours_wwv(pos: np.ndarray, species: np.ndarray) -> MethodResult:
    """canonicalize_3d_wwv -- the PCA-free proposal. Same reindexing note as
    run_ours."""
    pos = np.asarray(pos, dtype=float)
    species = np.asarray(species)
    try:
        res = canonicalize_3d_wwv(pos)
        species_out = species[res.order]
        if not np.all(np.isfinite(res.points)):
            return MethodResult(None, None, True, "InvalidOutput")
        return MethodResult(res.points, species_out, False, None, res.regime, res.n_candidates)
    except Exception as e:  # noqa: BLE001
        return MethodResult(None, None, True, type(e).__name__)


METHODS = {
    "asun": run_asun,
    "pca": run_pca,
    "ours": run_ours,
    "ours_wwv": run_ours_wwv,
}


if __name__ == "__main__":
    # Tiny smoke test: NH3-like tetrahedral-ish cloud, rotate, check all
    # methods run and (ours/PCA) are rotation-consistent up to floating noise.
    from scipy.spatial.transform import Rotation

    rng = np.random.RandomState(0)
    pos = np.array(
        [[0, 0, 0.3], [1, 0, -0.1], [-0.5, 0.866, -0.1], [-0.5, -0.866, -0.1]],
        dtype=float,
    )
    species = np.array([7, 1, 1, 1])
    Q = Rotation.random(random_state=0).as_matrix()
    pos2 = pos @ Q.T

    for name, fn in METHODS.items():
        r1 = fn(pos.copy(), species.copy())
        r2 = fn(pos2.copy(), species.copy())
        ok1 = "CRASH:" + r1.err_type if r1.crashed else "ok"
        ok2 = "CRASH:" + r2.err_type if r2.crashed else "ok"
        print(f"{name:10s} baseline={ok1:16s} rotated={ok2:16s}")
