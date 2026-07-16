"""
Noise scaling: how a perturbation of the input propagates to the output.

For a canonicalizer C and a clean cloud X, we add isotropic Gaussian jitter eps
to the input and measure how much the *output* moves relative to how much the
*input* moved:

    input noise   = per-point RMS of eps                    (how far we pushed X)
    output noise  = per-point RMS displacement between       (how far C moved)
                    C(X) and C(X + eps), under the optimal
                    point-to-point matching
    amplification = output noise / input noise

A stable canonicalizer just carries the noise through a rotation/reordering, so
amplification ~ 1 (a rotation preserves per-point norm; the matching absorbs any
benign row reshuffle, and symmetry-equivalent frame flips leave the output *set*
unchanged). Amplification >> 1 means a small input nudge made the algorithm jump
to a genuinely different canonical representative -- a frame flip at a near-tie,
a degenerate eigenframe, an anchor swap. That is the discontinuity firing, and
it is exactly what distinguishes a robust canonicalizer from a brittle one.

We sweep this for every shape in ``src.shapes.build_cases`` and compare:

  ours   canonical_radial (src/sign_invariance.py) -- geometry only.
  PCA    canonical_pca (src/sign_invariance.py) -- the eigenframe baseline
         (Gram eigenvectors + lexicographic sign/permutation fix). Species-
         blind. Its frame is a *continuous* eigendecomposition, so it is
         smooth where the spectrum is non-degenerate but ill-conditioned on
         isotropic/degenerate clouds (the eigenbasis is arbitrary there).
  ASUN   Baker et al. 2024 CategoricalPointCloud.CatFrame(tol=1e-2), vendored
         under results_reproduction/.../vendor/pyorbit. Fed a uniform species
         so it runs purely on geometry too -- the fair, species-blind match to
         ours. ASUN only rotates (never reorders); the matching handles that.

Both algorithms see the *same* noisy, re-centred input at each draw, so the
comparison is apples-to-apples. Per shape we plot, on log-log axes, the median
amplification (solid) and the p95 worst case (dashed) versus the input noise
level, one colour per algorithm, with a reference line at amplification = 1.

Usage
-----
    python -m experiments.noise_scaling                  # full sweep -> plots + csv
    python -m experiments.noise_scaling --reps 40
    python -m experiments.noise_scaling --smin 1e-6 --smax 3e-1 --nsigma 11
"""

import argparse
import csv
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment

from src.sign_invariance import canonical_radial, canonical_pca
from src.shapes import build_cases

# ASUN (vendored) pulls in torch, which is built against an older numpy; the
# import spams a harmless "compiled with numpy 1.x" warning we silence here.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    _ASUN = os.path.join(
        _ROOT,
        "results_reproduction",
        "experiments",
        "asun_robustness",
        "vendor",
        "pyorbit",
    )
    import sys

    if _ASUN not in sys.path:
        sys.path.insert(0, _ASUN)
    try:
        from CategoricalPointCloud import CatFrame  # noqa: E402

        _ASUN_OK = True
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] could not import ASUN ({type(_e).__name__}: {_e}); "
              "plotting ours only.")
        _ASUN_OK = False

ASUN_TOL = 1e-2  # ASUN's shipped default -- never tuned.

OURS_COLOR = "#3b6fd6"   # blue  (repo convention)
PCA_COLOR = "#009e73"    # bluish-green (Okabe-Ito; CVD-safe with blue+orange)
ASUN_COLOR = "#e08214"   # orange (colourblind-safe pair with the blue)
REF_COLOR = "0.55"

# algo -> (colour, marker, legend label). Order here is the plotting order.
STYLE = {
    "ours": (OURS_COLOR, "o", "ours (canonical_radial)"),
    "pca": (PCA_COLOR, "^", "PCA (canonical_pca)"),
    "asun": (ASUN_COLOR, "s", "ASUN"),
}


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def rms_radius(X):
    return float(np.sqrt(np.mean(np.sum(X**2, axis=1))))


def center(X):
    return X - X.mean(axis=0)


def per_point_rms(V):
    """RMS length of the rows of V (a per-point displacement field)."""
    return float(np.sqrt(np.mean(np.sum(V**2, axis=1))))


def matched_rms(A, B):
    """
    Per-point RMS displacement between clouds A and B under the optimal
    point-to-point assignment (species-blind Hungarian on Euclidean distance).

    Matching, not raw row order, is what makes this a fair "how far did the
    output move" measure: a benign row reshuffle costs nothing, while a genuine
    frame flip (the whole cloud rotated/reflected to a different representative)
    still shows up as large displacement.
    """
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    r, c = linear_sum_assignment(D)
    return float(np.sqrt(np.mean(D[r, c] ** 2)))


# --------------------------------------------------------------------------- #
# The two canonicalizers, behind a uniform (cloud -> pose or None) interface
# --------------------------------------------------------------------------- #
def canon_ours(X):
    return canonical_radial(center(X))


def canon_pca(X):
    return canonical_pca(center(X))


def canon_asun(X):
    """ASUN pose, or None if it crashes (a crash is a real ASUN outcome, not a
    zero -- it must never be silently scored as a perfect match)."""
    if not _ASUN_OK:
        return None
    Xc = center(X)
    species = np.ones(len(Xc), dtype=int)  # uniform: fair, species-blind ASUN
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aligned, _frame = CatFrame(tol=ASUN_TOL).get_frame(Xc.copy(), species)
        aligned = np.asarray(aligned, dtype=float)
        if aligned.shape != Xc.shape or not np.all(np.isfinite(aligned)):
            return None
        return aligned
    except Exception:  # noqa: BLE001 -- any failure is "no pose"
        return None


CANONS = {"ours": canon_ours, "pca": canon_pca, "asun": canon_asun}


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_shape(X, sigmas, reps, rng):
    """
    Returns dict algo -> (nsigma, reps) array of amplification ratios (np.nan
    where that algorithm crashed on either the clean or the noisy cloud).
    """
    n = len(X)
    scale = rms_radius(center(X))
    ratios = {a: np.full((len(sigmas), reps), np.nan) for a in CANONS}

    for j in range(reps):
        # One random pose per rep, shared by the clean reference and every
        # noisy draw, so the ratio isolates the noise (not the pose).
        Q = _random_rotation(rng)
        base = center(X @ Q.T)
        clean = {a: fn(base) for a, fn in CANONS.items()}

        for i, sigma in enumerate(sigmas):
            eps = sigma * scale * rng.normal(size=X.shape)
            in_rms = per_point_rms(eps)
            if in_rms == 0.0:
                continue
            noisy = base + eps
            for a, fn in CANONS.items():
                if clean[a] is None:
                    continue
                Y = fn(noisy)
                if Y is None:
                    continue
                ratios[a][i, j] = matched_rms(clean[a], Y) / in_rms
    return ratios


def _random_rotation(rng):
    """A uniformly random 3x3 rotation (QR of a Gaussian, sign-fixed)."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def summarise(ratios):
    """(nsigma, reps) ratios -> dict of per-sigma statistics, nan-aware."""
    out = {}
    for a, M in ratios.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[a] = {
                "median": np.nanmedian(M, axis=1),
                "p25": np.nanpercentile(M, 25, axis=1),
                "p75": np.nanpercentile(M, 75, axis=1),
                "p95": np.nanpercentile(M, 95, axis=1),
                "max": np.nanmax(M, axis=1),
                "n_ok": np.sum(np.isfinite(M), axis=1),
            }
    return out


def sweep(cases, sigmas, reps, seed):
    rng = np.random.default_rng(seed)
    results = {}
    for name, X in cases.items():
        stats = summarise(run_shape(np.asarray(X, float), sigmas, reps, rng))
        results[name] = stats
        parts = []
        for a in CANONS:
            med = stats[a]["median"]
            mx = stats[a]["max"]
            ok = stats[a]["n_ok"]
            if np.all(ok == 0):
                parts.append(f"{a}=crashed")
            else:
                parts.append(
                    f"{a}: med<={np.nanmax(med):.2g} max<={np.nanmax(mx):.2g}"
                )
        print(f"{name:15s} n={len(X):2d}  " + "  ".join(parts))
    return results


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _legend_handles(methods):
    """Compact proxy legend: one entry per method (colour + marker), plus two
    grey entries explaining the solid/dashed statistic split."""
    h = [Line2D([], [], color=STYLE[a][0], marker=STYLE[a][1], ms=5, lw=1.6,
                label=STYLE[a][2]) for a in methods]
    h.append(Line2D([], [], color="0.4", lw=1.6, ls="-", label="median"))
    h.append(Line2D([], [], color="0.4", lw=1.2, ls="--", label="p95 worst case"))
    return h


def _plot_one(ax, sigmas, stats, title, methods):
    plotted = False
    for a in methods:
        color, marker, _label = STYLE[a]
        if a not in stats:
            continue
        s = stats[a]
        if np.all(s["n_ok"] == 0):
            continue
        med = np.where(s["n_ok"] > 0, s["median"], np.nan)
        p95 = np.where(s["n_ok"] > 0, s["p95"], np.nan)
        ax.fill_between(sigmas, s["p25"], s["p75"], color=color, alpha=0.13,
                        linewidth=0)
        ax.loglog(sigmas, med, "-", color=color, marker=marker, ms=4)
        ax.loglog(sigmas, p95, "--", color=color, lw=1.2, alpha=0.9)
        plotted = True

    ax.axhline(1.0, color=REF_COLOR, lw=1.0, ls=":", zorder=0)
    # Flag any plotted method that never produced a pose for this shape.
    crashed = [STYLE[a][2].split(" (")[0] for a in methods
               if a in stats and np.all(stats[a]["n_ok"] == 0)]
    if crashed:
        ax.text(0.5, 0.5, f"{', '.join(crashed)}: crashed",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color=ASUN_COLOR,
                bbox=dict(boxstyle="round", fc="white", ec=ASUN_COLOR, alpha=0.8))
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.15)
    return plotted


def plot_grid(results, sigmas, path, methods):
    names = list(results)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        _plot_one(ax, sigmas, results[name], name, methods)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    axes[0, 0].legend(handles=_legend_handles(methods), fontsize=7.5,
                      loc="upper left")
    fig.suptitle(
        "Noise amplification of the canonicalizer  —  output noise / input noise\n"
        "(1 = pass-through, >1 = frame flip;  solid = median, dashed = p95 worst case)",
        fontsize=12,
    )
    fig.supxlabel("input noise sigma (fraction of cloud scale)")
    fig.supylabel("amplification  =  output RMS displacement / input RMS")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_each(results, sigmas, subdir, methods):
    os.makedirs(subdir, exist_ok=True)
    for name, stats in results.items():
        fig, ax = plt.subplots(figsize=(5.5, 4))
        _plot_one(ax, sigmas, stats, name, methods)
        ax.legend(handles=_legend_handles(methods), fontsize=8, loc="best")
        ax.set_xlabel("input noise sigma (fraction of cloud scale)")
        ax.set_ylabel("amplification (output / input RMS)")
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f"{name}.png"), dpi=130)
        plt.close(fig)


# Grid + per-shape figures for two method subsets: the full three-way
# comparison and the ours-vs-ASUN view with PCA removed.
VARIANTS = {
    "with_pca": ["ours", "pca", "asun"],
    "ours_vs_asun": ["ours", "asun"],
}


def write_summary(results, sigmas, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "algo", "sigma", "median_amp", "p95_amp",
                    "max_amp", "n_ok"])
        for name, stats in results.items():
            for a, s in stats.items():
                for i, sigma in enumerate(sigmas):
                    w.writerow([
                        name, a, f"{sigma:.3e}",
                        f"{s['median'][i]:.4g}",
                        f"{s['p95'][i]:.4g}",
                        f"{s['max'][i]:.4g}",
                        int(s["n_ok"][i]),
                    ])


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=24,
                        help="noise draws (and random poses) per sigma")
    parser.add_argument("--nsigma", type=int, default=11)
    parser.add_argument("--smin", type=float, default=1e-6)
    parser.add_argument("--smax", type=float, default=3e-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="viz_output/noise_scaling")
    args = parser.parse_args()

    cases = build_cases()
    sigmas = np.logspace(np.log10(args.smin), np.log10(args.smax), args.nsigma)

    os.makedirs(args.outdir, exist_ok=True)
    results = sweep(cases, sigmas, args.reps, args.seed)

    summary = os.path.join(args.outdir, "summary.csv")
    write_summary(results, sigmas, summary)
    print(f"\nwrote {summary}")
    for tag, methods in VARIANTS.items():
        grid = os.path.join(args.outdir, f"amplification_grid__{tag}.png")
        sub = os.path.join(args.outdir, f"per_shape__{tag}")
        plot_grid(results, sigmas, grid, methods)
        plot_each(results, sigmas, sub, methods)
        print(f"wrote {grid}")
        print(f"wrote {sub}/<shape>.png")


if __name__ == "__main__":
    main()
