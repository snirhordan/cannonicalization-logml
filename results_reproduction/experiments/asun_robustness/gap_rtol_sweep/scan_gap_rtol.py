"""Quick scan of candidate gap_rtol values to find one that routes the
delta in {0.01, 0.003, 0} band to AXIAL robustly under noise, without
over-routing delta=0.1 / 0.3 (well-separated) to AXIAL.
"""
import sys

from noise_robustness_test import run_sweep, format_table

for g in [1e-3, 3e-3, 5e-3, 1e-2, 2e-2]:
    res = run_sweep(g)
    print(format_table(res, g))
    print()
