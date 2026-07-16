"""Adversarial noise-robustness test of canonicalize_pca_full's dispatch
gap_rtol constant.

Question: as the covariance-eigenvalue gap between the two smallest
eigenvalues shrinks from well-separated toward degenerate (an ellipsoidal
cloud with semi-axes a1=1, a2=a1*(1+delta), a3=2.5), does the module route
to the stable AXIAL handler (canonicalize_in_eigenframe) instead of the
eigenvector-unstable DISTINCT handler (canonical_pca) -- and does the
*regime choice itself* stay stable under small coordinate noise, or does
noise flip the dispatch decision back and forth across the gap_rtol
threshold (which would show up as a large frame-flip in the final
canonical points, since DISTINCT and AXIAL construct the frame completely
differently)?

Method per (delta, sigma):
  - Build a fixed base ellipsoid cloud X0 (deterministic Fibonacci-sphere
    directions scaled by the three semi-axes) so the CLEAN gap ratio is
    reproducible and not itself a random variable.
  - For each of N_TRIALS trials: add iid Gaussian coordinate noise of scale
    sigma (relative to a1) to X0, apply a random O(3) rotation and a random
    translation, and canonicalize both the clean X0 and the noisy/rotated
    cloud with canonicalize_pca_full at a given gap_rtol.
  - Compare the two canonical point sets via an optimal-assignment (EMD-like)
    normalized distance, and separately record whether the dispatch REGIME
    string differs between the clean and noisy runs.
  - A "flip" is declared when the normalized EMD distance exceeds
    FLIP_THRESHOLD -- an O(1) discrepancy that cannot be explained by O(sigma)
    coordinate jitter alone (sigma <= 1e-2 throughout), so it must come from
    the two runs landing in genuinely different candidate frames/branches.

Run:  /home/snirhordan/miniconda3/envs/gnnplus/bin/python3 noise_robustness_test.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group

_CODE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "code"))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from canonicalize_pca_full import canonicalize_pca_full  # noqa: E402

RNG_MASTER_SEED = 20260716
N_TRIALS = 50
N_POINTS = 40
A1 = 1.0
A3 = 2.5  # third semi-axis, kept well separated from a1/a2 for all delta
DELTAS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.0]
SIGMAS = [0.0, 1e-4, 1e-3, 1e-2]
FLIP_THRESHOLD = 0.25  # normalized EMD distance (relative to A1) => "flip"


def fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic, near-uniform point set on the unit sphere. NOTE: this
    lattice is close to centrosymmetric (for every point ~ -point is also
    close to a lattice point), which makes canonical_pca's per-axis SIGN
    argmin sit near a tie -- a confound unrelated to the eigenvalue-gap
    question this test targets. Kept here only for reference; the cloud
    actually used is `_BASE_RAW` below (generic, not symmetric)."""
    i = np.arange(n, dtype=float)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


# Base cloud must be (a) GENERIC -- no near-centrosymmetric sign tie in
# canonical_pca's 2^3 sign argmin, so instability we measure is attributable
# to the eigen-gap dispatch question and not to a confounding sign-ambiguity
# near-tie (a symmetric lattice, e.g. Fibonacci-sphere directions, is close
# to centrosymmetric and confounds this) -- and (b) have an EXACTLY
# controlled clean eigenvalue gap, so the delta sweep isn't swamped by the
# O(1/sqrt(n)) sample-covariance fluctuation of a finite random point set
# (with n=40 raw Gaussian points that floor sits around gap/top ~ 5%,
# masking every delta <= 0.1). We get both by whitening a fixed generic
# random cloud in ITS OWN principal frame (a diagonal rescale in that frame
# cannot re-symmetrize the generic higher-order structure that keeps
# canonical_pca's sign choice unambiguous) and then rescaling the whitened
# columns to the EXACT target semi-axes (a1, a1*(1+delta), a3): the
# resulting sample covariance is diag(a1^2, a2^2, a3^2) to float precision,
# by construction, independent of delta.
_BASE_SEED = np.random.default_rng(31415926)
_BASE_RAW = _BASE_SEED.standard_normal((N_POINTS, 3))
_BASE_CENTERED = _BASE_RAW - _BASE_RAW.mean(axis=0)
_lam0, _V0 = np.linalg.eigh(_BASE_CENTERED.T @ _BASE_CENTERED)
_WHITE = (_BASE_CENTERED @ _V0) / np.sqrt(_lam0)[None, :]  # exact unit covariance


def ellipsoid_cloud(delta: float) -> np.ndarray:
    a2 = A1 * (1.0 + delta)
    axes = np.array([A1, a2, A3])
    return _WHITE * axes[None, :]


def clean_gap_ratios(X0: np.ndarray) -> tuple[float, float]:
    Xc = X0 - X0.mean(axis=0)
    lam = np.linalg.eigvalsh(Xc.T @ Xc)
    top = lam[-1]
    return float((lam[1] - lam[0]) / top), float((lam[2] - lam[1]) / top)


def normalized_emd(P: np.ndarray, Q: np.ndarray, scale: float) -> float:
    """Optimal-assignment (Hungarian) mean distance between two same-size
    point sets, normalized by `scale`. This is the discrete-uniform-mass
    Earth Mover's Distance and is agnostic to any row-order convention
    either handler used internally.
    """
    n = P.shape[0]
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    row, col = linear_sum_assignment(D)
    return float(D[row, col].mean() / scale)


def run_sweep(gap_rtol: float, rng_seed: int = RNG_MASTER_SEED) -> dict:
    results = {}
    for delta in DELTAS:
        X0 = ellipsoid_cloud(delta)
        gap01, gap12 = clean_gap_ratios(X0)
        canon0 = canonicalize_pca_full(X0, gap_rtol=gap_rtol)
        rng = np.random.default_rng((rng_seed, round(delta * 1e6)))

        per_sigma = {}
        for sigma in SIGMAS:
            emds = []
            regimes1 = []
            for _ in range(N_TRIALS):
                noise = sigma * A1 * rng.standard_normal(X0.shape)
                Xn = X0 + noise
                Q = ortho_group.rvs(dim=3, random_state=rng)
                t = rng.uniform(-3.0, 3.0, size=3)
                Xt = (Xn @ Q.T) + t

                canon1 = canonicalize_pca_full(Xt, gap_rtol=gap_rtol)
                emd = normalized_emd(canon0.points, canon1.points, A1)
                emds.append(emd)
                regimes1.append(canon1.regime)

            emds = np.array(emds)
            flip_rate = float(np.mean(emds > FLIP_THRESHOLD))
            regime_mismatch_rate = float(
                np.mean([r != canon0.regime for r in regimes1])
            )
            per_sigma[sigma] = {
                "mean_emd": float(emds.mean()),
                "median_emd": float(np.median(emds)),
                "max_emd": float(emds.max()),
                "flip_rate": flip_rate,
                "regime_mismatch_rate": regime_mismatch_rate,
                "regimes1_seen": sorted(set(regimes1)),
            }

        results[delta] = {
            "gap01_ratio_clean": gap01,
            "gap12_ratio_clean": gap12,
            "regime0": canon0.regime,
            "n_candidates0": canon0.n_candidates,
            "per_sigma": per_sigma,
        }
    return results


def format_table(results: dict, gap_rtol: float) -> str:
    lines = []
    lines.append(f"gap_rtol = {gap_rtol:g}")
    header = (
        f"{'delta':>8} {'regime0':>8} {'gap01/top':>12}"
        + "".join(f"  sigma={s:g}(mismatch%,meanEMD,maxEMD)".rjust(36) for s in SIGMAS)
    )
    lines.append(header)
    for delta in DELTAS:
        r = results[delta]
        row = f"{delta:>8.4f} {r['regime0']:>8} {r['gap01_ratio_clean']:>12.2e}"
        for s in SIGMAS:
            ps = r["per_sigma"][s]
            cell = (
                f"  {ps['regime_mismatch_rate']*100:5.1f}%,"
                f"{ps['mean_emd']:.2e},{ps['max_emd']:.2e}"
            )
            row += cell.rjust(36)
        lines.append(row)
    return "\n".join(lines)


OLD_GAP_RTOL = 1e-3   # module's original DEFAULT_GAP_RTOL, for comparison
NEW_GAP_RTOL = 2e-2   # tuned value now live in canonicalize_pca_full.py

if __name__ == "__main__":
    from canonicalize_pca_full import DEFAULT_GAP_RTOL

    assert DEFAULT_GAP_RTOL == NEW_GAP_RTOL, (
        f"module's DEFAULT_GAP_RTOL={DEFAULT_GAP_RTOL} != NEW_GAP_RTOL="
        f"{NEW_GAP_RTOL} -- update this script's NEW_GAP_RTOL to match."
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== BEFORE (original DEFAULT_GAP_RTOL = 1e-3) ===")
    res_before = run_sweep(OLD_GAP_RTOL)
    table_before = format_table(res_before, OLD_GAP_RTOL)
    print(table_before)
    with open(os.path.join(out_dir, "results_before.json"), "w") as f:
        json.dump({str(k): v for k, v in res_before.items()}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "table_before.txt"), "w") as f:
        f.write(table_before + "\n")

    print()
    print("=== AFTER (tuned DEFAULT_GAP_RTOL = 2e-2, module as shipped) ===")
    res_after = run_sweep(DEFAULT_GAP_RTOL)
    table_after = format_table(res_after, DEFAULT_GAP_RTOL)
    print(table_after)
    with open(os.path.join(out_dir, "results_after.json"), "w") as f:
        json.dump({str(k): v for k, v in res_after.items()}, f, indent=2, default=str)
    with open(os.path.join(out_dir, "table_after.txt"), "w") as f:
        f.write(table_after + "\n")

    # Sanity check the task explicitly calls out: a genuinely well-separated
    # cloud (delta=0.3) must still dispatch to DISTINCT (canonical_pca), both
    # before and after the retune, at every noise level tested.
    for tag, res in (("BEFORE", res_before), ("AFTER", res_after)):
        r03 = res[0.3]
        assert r03["regime0"] == "DISTINCT", f"{tag}: delta=0.3 regime0 != DISTINCT"
        for s in SIGMAS:
            seen = r03["per_sigma"][s]["regimes1_seen"]
            assert seen == ["DISTINCT"], (
                f"{tag}: delta=0.3, sigma={s} routed to {seen}, expected only DISTINCT"
            )
    print()
    print("Sanity check passed: delta=0.3 (well-separated) stayed DISTINCT "
          "at every noise level, before and after retuning gap_rtol.")
