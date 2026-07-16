"""metrics.py -- EMD (species-constrained Hungarian) + frame-residual
Procrustes angle + flip-rate, exactly as specified in the harness brief.

consistency_metrics(A, species_A, B, species_B) compares two canonical
poses A, B (each (n,3), same n, same species multiset) that are supposed to
represent the SAME underlying orbit (e.g. canon(baseline) vs
canon(rotated+noised copy)):

  1. Species-constrained optimal assignment: cost[i,j] = ||A[i]-B[j]|| if
     species_A[i] == species_B[j] else +inf, solved by
     scipy.optimize.linear_sum_assignment. Since every atom has a
     same-species partner (both clouds are the same molecule), a finite
     perfect matching always exists.
  2. EMD = mean matched distance (this IS exact Wasserstein-1 under the
     assignment metric since the two point sets have equal, matched mass).
     NOTE: EMD has a sigma floor -- even a perfect canonicalizer gives
     EMD = O(sigma) at sigma>0, because the noise itself displaces the
     matched points. Report it as such; do not expect it to vanish for
     sigma>0.
  3. Frame-residual angle: after matching, run orthogonal Procrustes
     (scipy.linalg.orthogonal_procrustes) between the matched rows of A and
     B to get the best-fit rotation R*, then
         theta = arccos(clip((trace(R*) - 1) / 2, -1, 1))   in degrees.
     Procrustes removes any residual global rotation between the two
     canonical poses, so theta isolates FRAME instability (e.g. an ASUN
     axis/anchor flip) from ordinary noise-induced jitter. It has NO sigma
     floor for a rotation-invariant method: at sigma=0 a consistent method
     gives theta ~ 0 exactly, and small sigma should give small theta unless
     the frame itself jumped.
  4. FLIP-RATE (computed by the caller across many perturbations): fraction
     of perturbations with theta > 5 degrees.

All three quantities are computed only when both A and B are present
(non-crashed); crashes are the caller's responsibility to track separately
and must NEVER be folded into these means as 0 or skipped silently.

KNOWN BLIND SPOT (found empirically during pilot validation, documented
here rather than silently patched): theta is ILL-CONDITIONED on RANK-
DEFICIENT point clouds -- most importantly, EXACTLY LINEAR molecules
(D*h/C*v: CO2, acetylene, and any QM9 molecule pymatgen labels as linear).
For a perfectly collinear cloud, both A and B have zero extent in the two
directions perpendicular to the molecular axis, so orthogonal_procrustes's
best-fit rotation in THOSE two directions is a pure numerical-noise
artifact (the SVD has no real signal to align there) -- it can report
theta up to 90-180 degrees even when the two poses are, in every
meaningful sense (matched positions, EMD), essentially identical. We
verified this directly: canonicalize_3d on synthetic CO2 gives EMD ~ 2e-16
between a clean and a purely-rotated (sigma=0) copy, yet theta = 90 deg,
purely from Procrustes gauge freedom on the degenerate axes. FOR LINEAR
(D*h/C*v) STRATA, TRUST EMD, NOT THETA, AS THE CONSISTENCY SIGNAL. Every
other stratum tested (all polygons/bipyramids/prisms/Platonic solids/real
molecules -- i.e. anything with 3D extent, rank 3) does not have this
issue and theta is fully meaningful there.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes


class ConsistencyResult(NamedTuple):
    emd: float
    theta_deg: float
    n_matched: int


def species_constrained_match(A: np.ndarray, species_A: np.ndarray,
                               B: np.ndarray, species_B: np.ndarray):
    """Return (row_idx, col_idx) of the optimal species-constrained
    assignment minimizing total Euclidean distance. Raises ValueError if no
    finite perfect matching exists (species multisets differ)."""
    n = A.shape[0]
    assert B.shape[0] == n, f"shape mismatch: {A.shape} vs {B.shape}"
    diff = A[:, None, :] - B[None, :, :]                # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)                  # (n, n)
    mismatch = species_A[:, None] != species_B[None, :]
    cost = np.where(mismatch, np.inf, dist)
    row_idx, col_idx = linear_sum_assignment(cost)
    matched_cost = cost[row_idx, col_idx]
    if not np.all(np.isfinite(matched_cost)):
        raise ValueError(
            "no finite species-constrained perfect matching -- species "
            "multisets of A and B must be identical"
        )
    return row_idx, col_idx


def frame_residual_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Orthogonal Procrustes rotation angle (degrees) between two ALREADY
    matched (same row order, same n) point sets. A and B should already be
    centered (canonical poses from all four methods are centered), but we
    re-center defensively since Procrustes assumes it."""
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    R, _scale = orthogonal_procrustes(Bc, Ac)  # R minimizes ||Bc @ R - Ac||
    # orthogonal_procrustes returns R s.t. Bc @ R ~= Ac; R is orthogonal.
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))
    return float(theta)


def consistency_metrics(A: np.ndarray, species_A: np.ndarray,
                         B: np.ndarray, species_B: np.ndarray) -> ConsistencyResult:
    row_idx, col_idx = species_constrained_match(A, species_A, B, species_B)
    A_m = A[row_idx]
    B_m = B[col_idx]
    d = np.linalg.norm(A_m - B_m, axis=1)
    emd = float(d.mean())
    theta = frame_residual_angle_deg(A_m, B_m)
    return ConsistencyResult(emd=emd, theta_deg=theta, n_matched=len(row_idx))


FLIP_THRESHOLD_DEG = 5.0


def is_flip(theta_deg: float) -> bool:
    return theta_deg > FLIP_THRESHOLD_DEG


if __name__ == "__main__":
    # Self-test: sigma=0 consistency for all four methods on a random cloud.
    # Every non-crashing method must give EMD ~ 0 and theta ~ 0 (assert
    # < 1e-6 for ours/PCA; ASUN should also be ~0 on smooth, non-symmetric
    # inputs -- print results either way).
    import os
    import sys
    from scipy.spatial.transform import Rotation

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from methods import METHODS  # noqa: E402

    rng = np.random.RandomState(42)
    n = 9
    pos = rng.randn(n, 3)
    pos -= pos.mean(axis=0)
    species = rng.choice([1, 6, 7, 8], size=n)

    Q = Rotation.random(random_state=7).as_matrix()
    pos_rot = pos @ Q.T  # sigma=0: pure rotation, no noise

    print(f"{'method':10s} {'crash_base':10s} {'crash_rot':10s} {'EMD':>12s} {'theta_deg':>12s}")
    failures = []
    for name, fn in METHODS.items():
        r_base = fn(pos.copy(), species.copy())
        r_rot = fn(pos_rot.copy(), species.copy())
        if r_base.crashed or r_rot.crashed:
            print(f"{name:10s} {str(r_base.crashed):10s} {str(r_rot.crashed):10s} "
                  f"{'--':>12s} {'--':>12s}  (err: {r_base.err_type}, {r_rot.err_type})")
            continue
        res = consistency_metrics(r_base.pose, r_base.species_out,
                                   r_rot.pose, r_rot.species_out)
        print(f"{name:10s} {'False':10s} {'False':10s} {res.emd:12.3e} {res.theta_deg:12.3e}")
        if name in ("ours", "ours_wwv", "pca"):
            try:
                # EMD is linear in position error, so 1e-6 (spec value) is
                # the right bound and is met with room to spare (~1e-15).
                assert res.emd < 1e-6, f"{name}: EMD {res.emd} >= 1e-6 at sigma=0"
                # theta = arccos((tr-1)/2) has derivative -> infinity as
                # theta -> 0 (cos(theta) ~ 1 - theta^2/2), so a trace error
                # of O(machine eps ~ 2e-16) -- unavoidable in an SVD-based
                # orthogonal Procrustes -- is amplified to
                # theta ~ sqrt(2*eps) ~ 1e-8 rad ~ 1e-6 deg *at best*, and
                # empirically ~1e-6 to a few 1e-6 deg here. A literal
                # "< 1e-6 deg" bound is therefore at or below the float64
                # noise floor of THIS formula (verified empirically: ours
                # gave 1.7e-6, ours_wwv 2.4e-6 on the first random seed we
                # tried) and would fail for ANY rotation-invariant method,
                # not just ours. We use 1e-4 deg instead: still 4-5 orders
                # of magnitude below FLIP_THRESHOLD_DEG=5, so it does not
                # weaken flip detection at all.
                assert res.theta_deg < 1e-4, f"{name}: theta {res.theta_deg} >= 1e-4 at sigma=0"
            except AssertionError as e:
                failures.append(str(e))
        elif name == "asun":
            if res.emd >= 1e-6 or res.theta_deg >= 1e-4:
                print(f"  [note] ASUN not < (1e-6 EMD, 1e-4 deg) at sigma=0 on this smooth input "
                      f"(EMD={res.emd:.3e}, theta={res.theta_deg:.3e}) -- not asserted, just reported")

    if failures:
        print("\nSELF-TEST FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("\nSELF-TEST PASSED: ours/ours_wwv/pca all give EMD<1e-6 and theta<1e-4 deg at sigma=0"
              " (theta bound relaxed from the nominal 1e-6 deg -- see comment above; "
              "still 4-5 orders of magnitude under the 5 deg flip threshold).")
