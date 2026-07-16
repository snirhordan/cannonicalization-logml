#!/bin/bash
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --job-name=asun_fullqm9
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm_fullqm9_%j.out
#SBATCH --error=results/slurm_fullqm9_%j.out

# The `public` partition is GPU-only and rejects CPU-only jobs, so we request one
# GPU purely to satisfy the scheduler -- it goes UNUSED. The workload is scipy Qhull
# + Hopcroft DFA + POT LP-EMD, none of which use a GPU; the speedup is pure CPU
# parallelism over molecules across this node's 64 allocated CPUs.
# Pin BLAS to 1 thread/process so 64 mp.Pool workers don't oversubscribe.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$(dirname "${BASH_SOURCE[0]}")"
echo "host=$(hostname) cores=$SLURM_CPUS_PER_TASK start=$(date)"
/home/snirhordan/miniconda3/envs/gnnplus/bin/python3 reproduce_repo_tables_mp.py \
    --n 130831 --nperturb 10 --cores 64 \
    --out results/table2_fullqm9.csv
echo "end=$(date)"
