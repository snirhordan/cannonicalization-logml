#!/bin/bash
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --job-name=asun_subsets
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm_subsets_%j.out
#SBATCH --error=results/slurm_subsets_%j.out

# Test the subset hypothesis for ASUN's rank-3 mean: does a RANDOM 5000-molecule
# subset move it toward the paper's 0.028, and how much does it vary by subset?
# (GPU requested only to satisfy the GPU-only partition; unused. 64 CPUs do the work.)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "host=$(hostname) cores=$SLURM_CPUS_PER_TASK start=$(date)"
PY=/home/snirhordan/miniconda3/envs/gnnplus/bin/python3
for S in 1 2 3 4 5; do
  echo "========== RANDOM SUBSET seed=$S =========="
  $PY reproduce_repo_tables_mp.py --n 5000 --nperturb 100 --cores 64 \
      --subset-seed $S --out results/reproduce_rand5k_s${S}.csv 2>&1 \
      | grep -viE 'RuntimeWarning|arccos|invalid value|UserWarning|FutureWarning|deprecat'
done
echo "ALL SUBSETS DONE $(date)"
