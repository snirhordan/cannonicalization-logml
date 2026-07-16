#!/bin/bash
#SBATCH -p public
#SBATCH --account=cs
#SBATCH -c 64
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --job-name=asun_noise
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ASUN-robustness full sweep. CPU-only (no --gres/GPU requested, matching
# the verified account constraints: AllowAccounts=cs, --account=cs
# required, NOT dym-lab, NOT a GPU partition).
#
# DO NOT SUBMIT THIS DIRECTLY -- prepared for the orchestrator to `sbatch`
# after reviewing the pilot gate/effect results. It is idempotent to rerun
# (labeling and molecule-set caches are reused; run_sweep.py always
# (re)writes results/full.csv and results/full_raw_per_molecule.csv).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "[submit_full] host=$(hostname) date=$(date) cores=${SLURM_CPUS_PER_TASK:-64}"

conda run -n gnnplus python3 run_sweep.py \
    --mode full \
    --cores 64 \
    --out results/full.csv \
    --sigmas 0,1e-4,1e-3,3e-3,1e-2,3e-2,1e-1 \
    --nperturb 100 \
    --seed 0

echo "[submit_full] done: results/full.csv and results/full_raw_per_molecule.csv"
