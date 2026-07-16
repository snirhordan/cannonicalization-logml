#!/bin/bash
#SBATCH --partition=public
#SBATCH --account=cs
#SBATCH --job-name=asun_ae
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm_ae_%j.out
#SBATCH --error=results/slurm_ae_%j.out

# GPU requested only to satisfy the GPU-only partition; AE inference runs on CPU.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "host=$(hostname) start=$(date)"
/home/snirhordan/miniconda3/envs/gnnplus/bin/python3 run_ae_table2.py
echo "end=$(date)"
