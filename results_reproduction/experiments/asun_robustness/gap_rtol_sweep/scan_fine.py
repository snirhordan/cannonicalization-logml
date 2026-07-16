from noise_robustness_test import run_sweep, format_table

for g in [0.012, 0.015, 0.018, 0.02, 0.022, 0.025, 0.03]:
    res = run_sweep(g)
    print(format_table(res, g))
    print()
