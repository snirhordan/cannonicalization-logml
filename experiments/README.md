# Robustness experiments

`robustness.py` measures how stable `canonical_radial` is when the point cloud
is perturbed by small isotropic Gaussian noise *before* canonicalization, and
whether different views (O(3) + permutation) plus different noise draws still
land on the same canonical form.

## Run

```
python -m experiments.robustness                 # full sweep -> viz_output/robustness/
python -m experiments.robustness --reps 40       # more observations per condition
python -m experiments.robustness --ghost cube --ghost-sigma 1e-3   # visual overlay
```

Outputs land in `viz_output/robustness/`:
- `discrepancy_grid.png` — per cloud, aligned vs shape discrepancy vs noise.
- `flip_heatmap.png` — cloud × noise flip-rate heatmap (the headline).
- `summary.csv` — per-cloud invariance error, flip onset, C4/C3 ratio.
- ghost overlays (on demand) — many canonical outputs on one axis.

## Design (factorial)

Base clean cloud `X`; view `g = (Q in O(3), permutation)`; noise
`eps ~ sigma * scale * N(0, I)` with `scale` = RMS radius. Per (cloud, sigma):

| Condition | View | Noise | Question |
|-----------|------|-------|----------|
| **C2** | varied | fixed | Is *invariance* robust to noise? (should be ~1e-13) |
| **C3** | fixed | varied | Noise floor: wander from noise alone |
| **C4** | varied | varied | Realistic: different pose + independent noise |

## Metrics

Between two canonical outputs (both in canonical row order), normalised by the
cloud scale:
- **aligned** = `||Y1 - Y2||_F / (sqrt(n)*scale)` — sees frame/order flips.
- **shape** = `||sort(pdist Y1) - sort(pdist Y2)|| / (sqrt(#pairs)*scale)` —
  permutation/orientation-invariant; the true geometric change.
- **flip**: `aligned > 5*shape` and `aligned > 1e-6` — the frame jumped without
  the geometry changing (the discontinuity fired).

## How to read the results

- **aligned tracks shape** (two lines together, both ~linear in sigma) ⇒ the
  canonical form moves only as much as the geometry does: continuous, stable,
  Lipschitz. This is the *robust* signature (generic, tetrahedron, prism,
  chiral_C3, improper_S4).
- **aligned pinned near O(1) while shape stays tiny** ⇒ the frame flips between
  point-group-equivalent copies on essentially unchanged geometry. This is the
  *fragile* signature and appears for the highly symmetric clouds (cube,
  icosahedron, dodecahedron, cuboctahedron, sphere, ...). It is the
  mathematically correct behaviour at a bifurcation — a symmetric cloud's frame
  is genuinely ambiguous, so any symmetry-breaking noise resolves it randomly.

Note the C3 (fixed-view) curve sits on top of C4 (varied-view) for the fragile
clouds: the flips are driven by the **degeneracy**, not by the view. And for the
generic cloud the flip rate only rises at very large sigma — that is the
single-max-radius pole switching to a different point once noise exceeds the gap
between the two largest radii, the known fragility of a single-point pole.
