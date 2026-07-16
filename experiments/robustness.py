"""
Robustness of canonical_radial to small perturbations.

We add isotropic Gaussian jitter to each cloud *before* canonicalization and ask
whether different views (O(3) + permutation) and different noise draws still
land on the same canonical form. See the factorial design below.

For a base clean cloud X, a "view" g = (Q in O(3), permutation), noise
eps ~ sigma * scale * N(0, I) with scale = RMS radius, we run three conditions
per (cloud, sigma):

  C2  varied view, FIXED noise   -> canonical(g_j . (X+eps))
        Tests INVARIANCE under noise: outputs should be identical (~1e-13).
        Anything larger means a view resolved a noise-induced tie differently.

  C3  FIXED view, varied noise    -> canonical(g0 . X + eps_j)
        The NOISE FLOOR: how far the output wanders from noise alone.

  C4  varied view, varied noise   -> canonical(g_j . X + eps_j)
        The REALISTIC case (different pose + independent noise each observation).

Metrics between two canonical outputs (both already in canonical row order),
normalised by the cloud scale so they are dimensionless:

  aligned  = ||Y1 - Y2||_F / (sqrt(n) * scale)      (sees frame / order flips)
  shape    = ||sort(pdist Y1) - sort(pdist Y2)||     (permutation/orientation
             / (sqrt(#pairs) * scale)                 invariant; true geometry)

A pair "flips" when aligned > K * shape and aligned > FLOOR: the frame jumped
without the geometry changing -- the discontinuity fired. K = 5.

Usage
-----
    python -m experiments.robustness                     # full sweep -> plots + csv
    python -m experiments.robustness --reps 40
    python -m experiments.robustness --ghost cube --ghost-sigma 1e-3   # visual overlay
"""

import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.distance import pdist
from scipy.stats import ortho_group

from src.sign_invariance import canonical_radial
from src.shapes import build_cases

K_FLIP = 5.0          # aligned/shape ratio above which a pair counts as a flip
FLOOR = 1e-6          # absolute floor (normalised) so tiny-noise pairs never flip
ALIGNED_COLOR = "#3b6fd6"
SHAPE_COLOR = "#e08214"
FLOOR_COLOR = "0.55"


# --------------------------------------------------------------------------- #
# Core primitives
# --------------------------------------------------------------------------- #
def rms_radius(X):
    return float(np.sqrt(np.mean(np.sum(X**2, axis=1))))


def apply_view(X, Q, perm):
    return (X @ Q)[perm]


def random_view(rng, n):
    return ortho_group.rvs(dim=3, random_state=rng), rng.permutation(n)


def aligned_discrepancy(Y1, Y2, scale):
    return np.linalg.norm(Y1 - Y2) / (np.sqrt(len(Y1)) * scale)


def shape_key(Y):
    return np.sort(pdist(Y))


def shape_discrepancy(k1, k2, scale):
    if k1.size == 0:
        return 0.0
    return np.linalg.norm(k1 - k2) / (np.sqrt(k1.size) * scale)


def pairwise_stats(Ys, scale):
    """Median aligned/shape discrepancy and flip rate over all output pairs."""
    keys = [shape_key(Y) for Y in Ys]
    aligned, shape = [], []
    for i in range(len(Ys)):
        for j in range(i + 1, len(Ys)):
            aligned.append(aligned_discrepancy(Ys[i], Ys[j], scale))
            shape.append(shape_discrepancy(keys[i], keys[j], scale))
    aligned, shape = np.array(aligned), np.array(shape)
    flips = (aligned > K_FLIP * shape) & (aligned > FLOOR)
    return {
        "aligned": float(np.median(aligned)),
        "shape": float(np.median(shape)),
        "flip_rate": float(np.mean(flips)),
        "aligned_max": float(aligned.max()),
    }


# --------------------------------------------------------------------------- #
# The three conditions
# --------------------------------------------------------------------------- #
def run_conditions(X, sigma, reps, rng):
    n = len(X)
    scale = rms_radius(X)
    noise = lambda: sigma * scale * rng.normal(size=X.shape)

    # C2: one fixed noisy cloud, many views.
    N = X + noise()
    c2 = [canonical_radial(apply_view(N, *random_view(rng, n))) for _ in range(reps)]

    # C3: one fixed view, many independent noise draws.
    Q0, perm0 = random_view(rng, n)
    base = apply_view(X, Q0, perm0)
    c3 = [canonical_radial(base + noise()) for _ in range(reps)]

    # C4: independent view AND noise per observation.
    c4 = []
    for _ in range(reps):
        Q, perm = random_view(rng, n)
        c4.append(canonical_radial(apply_view(X, Q, perm) + noise()))

    return {
        "invariance_max": max(aligned_discrepancy(c2[0], y, scale) for y in c2[1:]),
        "C3": pairwise_stats(c3, scale),
        "C4": pairwise_stats(c4, scale),
    }


def sweep(cases, sigmas, reps, seed):
    rng = np.random.default_rng(seed)
    results = {}
    for name, X in cases.items():
        rows = [run_conditions(X, s, reps, rng) for s in sigmas]
        results[name] = rows
        print(
            f"{name:15s} n={len(X):2d}  "
            f"invariance<={max(r['invariance_max'] for r in rows):.1e}  "
            f"max flip_rate={max(r['C4']['flip_rate'] for r in rows):.2f}"
        )
    return results


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_discrepancy_grid(results, sigmas, path):
    names = list(results)
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    floor = 1e-16
    for ax, name in zip(axes.ravel(), names):
        rows = results[name]
        al4 = np.maximum([r["C4"]["aligned"] for r in rows], floor)
        sh4 = np.maximum([r["C4"]["shape"] for r in rows], floor)
        al3 = np.maximum([r["C3"]["aligned"] for r in rows], floor)
        ax.loglog(sigmas, al4, "-o", color=ALIGNED_COLOR, ms=4, label="aligned (C4)")
        ax.loglog(sigmas, sh4, "-s", color=SHAPE_COLOR, ms=4, label="shape (C4)")
        ax.loglog(sigmas, al3, "--", color=FLOOR_COLOR, label="aligned (C3, floor)")
        ax.set_title(name, fontsize=10)
        ax.grid(True, which="both", alpha=0.15)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.supxlabel("noise sigma (fraction of cloud scale)")
    fig.supylabel("normalised discrepancy   (aligned >> shape  =>  frame flip)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_flip_heatmap(results, sigmas, path):
    names = list(results)
    M = np.array([[r["C4"]["flip_rate"] for r in results[n]] for n in names])
    fig, ax = plt.subplots(figsize=(1.1 * len(sigmas) + 3, 0.45 * len(names) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{s:.0e}" for s in sigmas], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("noise sigma")
    ax.set_title("C4 flip rate (view + noise both varied)")
    fig.colorbar(im, ax=ax, label="fraction of output pairs that flip")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary(results, sigmas, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["cloud", "invariance_max", "flip_onset_sigma", "max_flip_rate", "C4_over_C3"]
        )
        for name, rows in results.items():
            inv = max(r["invariance_max"] for r in rows)
            onset = next(
                (s for s, r in zip(sigmas, rows) if r["C4"]["flip_rate"] > 0.5), np.nan
            )
            maxflip = max(r["C4"]["flip_rate"] for r in rows)
            ratios = [
                r["C4"]["aligned"] / r["C3"]["aligned"]
                for r in rows
                if r["C3"]["aligned"] > 0
            ]
            w.writerow(
                [
                    name,
                    f"{inv:.2e}",
                    f"{onset:.1e}" if np.isfinite(onset) else "none",
                    f"{maxflip:.2f}",
                    f"{np.median(ratios):.1f}" if ratios else "n/a",
                ]
            )


def ghost_overlay(name, X, sigma, reps, seed):
    """Overlay many C4 canonical outputs on one axis; flips show as ghosting."""
    rng = np.random.default_rng(seed)
    n = len(X)
    scale = rms_radius(X)
    outs = []
    for _ in range(reps):
        Q, perm = random_view(rng, n)
        eps = sigma * scale * rng.normal(size=X.shape)
        outs.append(canonical_radial(apply_view(X, Q, perm) + eps))
    limit = 1.15 * max(np.linalg.norm(Y, axis=1).max() for Y in outs)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    for Y in outs:
        ax.scatter(*Y.T, color=ALIGNED_COLOR, s=40, alpha=0.25, depthshade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.view_init(elev=20, azim=-60)
    ax.set_title(
        f"{name}: {reps} canonical outputs at sigma={sigma:.0e}\n"
        "crisp = stable, ghosted = flips"
    )
    return fig


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=24, help="observations per condition")
    parser.add_argument("--nsigma", type=int, default=8, help="number of noise levels")
    parser.add_argument("--smin", type=float, default=1e-8)
    parser.add_argument("--smax", type=float, default=1e-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="viz_output/robustness")
    parser.add_argument("--ghost", help="cloud name for a ghosting overlay")
    parser.add_argument("--ghost-sigma", type=float, default=1e-3)
    args = parser.parse_args()

    cases = build_cases()
    sigmas = np.logspace(np.log10(args.smin), np.log10(args.smax), args.nsigma)

    if args.ghost:
        if args.ghost not in cases:
            raise SystemExit(f"unknown cloud {args.ghost!r}; choices: {list(cases)}")
        ghost_overlay(args.ghost, cases[args.ghost], args.ghost_sigma, args.reps, args.seed)
        plt.show()
        return

    os.makedirs(args.outdir, exist_ok=True)
    results = sweep(cases, sigmas, args.reps, args.seed)

    grid = os.path.join(args.outdir, "discrepancy_grid.png")
    heat = os.path.join(args.outdir, "flip_heatmap.png")
    summary = os.path.join(args.outdir, "summary.csv")
    plot_discrepancy_grid(results, sigmas, grid)
    plot_flip_heatmap(results, sigmas, heat)
    write_summary(results, sigmas, summary)
    print(f"\nwrote {grid}\nwrote {heat}\nwrote {summary}")


if __name__ == "__main__":
    main()
