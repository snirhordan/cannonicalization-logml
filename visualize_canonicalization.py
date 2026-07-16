"""
Visualize canonical_radial before and after canonicalization.

For each test-case point cloud we show a top row of several random O(3)+
permutation variants of the same cloud and a bottom row of their
canonicalizations. If the algorithm works, every panel in the bottom row is
identical (same shape *and* same colour order, since the canonical row is
coloured by row index).

Usage
-----
    python visualize_canonicalization.py                 # save every case to viz_output/
    python visualize_canonicalization.py --show          # also open them interactively
    python visualize_canonicalization.py --case cube     # one case, interactive
    python visualize_canonicalization.py --list          # list case names

Dependencies: numpy, matplotlib, scipy.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables projection="3d")

from src.shapes import build_cases
from experiments.methods import METHODS

# CVD-safe pair: blue for the varied inputs, viridis for the canonical order.
INPUT_COLOR = "#3b6fd6"
VIEW = dict(elev=20, azim=-60)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _hull_edges(ax, X, color, lw):
    """Draw convex-hull edges to make solids legible; skip if degenerate."""
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(X)
        seen = set()
        for tri in hull.simplices:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                e = (min(a, b), max(a, b))
                if e in seen:
                    continue
                seen.add(e)
                ax.plot(*X[[a, b]].T, color=color, lw=lw, alpha=0.35)
    except Exception:
        pass  # coplanar / collinear / too few points -> just show markers


def _draw(ax, X, limit, color=None, by_index=False):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.view_init(**VIEW)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if by_index:
        c = np.arange(len(X))
        ax.scatter(*X.T, c=c, cmap="viridis", s=60, depthshade=False)
        _hull_edges(ax, X, "0.4", 0.8)
    else:
        ax.scatter(*X.T, color=color, s=60, depthshade=False)
        _hull_edges(ax, X, color, 0.8)


def plot_case(name, X, canon, n_variants=4, seed=1):
    rng = np.random.default_rng(seed)
    canon0 = canon(X)

    inputs, canons = [X], [canon0]
    for _ in range(n_variants - 1):
        Q = _random_orthogonal(rng)
        Z = X[rng.permutation(len(X))] @ Q
        inputs.append(Z)
        canons.append(canon(Z))

    limit = 1.15 * max(np.linalg.norm(c, axis=1).max() for c in inputs + canons)

    fig = plt.figure(figsize=(3.2 * n_variants, 6.6))
    fig.suptitle(f"{name}   (n = {len(X)})", fontsize=14, y=0.98)
    for k in range(n_variants):
        ax = fig.add_subplot(2, n_variants, k + 1, projection="3d")
        _draw(ax, inputs[k], limit, color=INPUT_COLOR)
        ax.set_title("original" if k == 0 else f"rotated + permuted #{k}", fontsize=9)

        ax = fig.add_subplot(2, n_variants, n_variants + k + 1, projection="3d")
        _draw(ax, canons[k], limit, by_index=True)
        ax.set_title("canonicalized", fontsize=9)

    fig.text(0.5, 0.52, "inputs (should differ)", ha="center", fontsize=10, color="0.3")
    fig.text(
        0.5, 0.04, "outputs (should all match)", ha="center", fontsize=10, color="0.3"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def _random_orthogonal(rng):
    from scipy.stats import ortho_group

    return ortho_group.rvs(dim=3, random_state=rng)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="render a single case interactively")
    parser.add_argument("--show", action="store_true", help="open figures interactively")
    parser.add_argument("--list", action="store_true", help="list case names and exit")
    parser.add_argument("--method", default="radial", choices=list(METHODS))
    parser.add_argument("--outdir", default=None, help="default: viz_output/<method>")
    parser.add_argument("--variants", type=int, default=4, help="orientations per case")
    args = parser.parse_args()

    canon = METHODS[args.method]
    outdir = args.outdir or os.path.join("viz_output", args.method)
    cases = build_cases()

    if args.list:
        print("\n".join(cases))
        return

    if args.case:
        if args.case not in cases:
            raise SystemExit(f"unknown case {args.case!r}; try --list")
        plot_case(args.case, cases[args.case], canon, n_variants=args.variants)
        plt.show()
        return

    os.makedirs(outdir, exist_ok=True)
    for name, X in cases.items():
        fig = plot_case(name, X, canon, n_variants=args.variants)
        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=130)
        print(f"wrote {path}")
        if not args.show:
            plt.close(fig)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
