#!/bin/bash
#SBATCH -J elf_hw_nl13
#SBATCH -A LRN105
#SBATCH -p service
#SBATCH -q develop
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
set -euo pipefail
cd /lustre/orion/proj-shared/lrn105/yangc/tva_elf_headwater_final/lstm_1layer_52site_nonlinear_heads_weekly_cosine
module load rocm/7.1.1 craype-accel-amd-gfx90a
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export MIOPEN_USER_DB_PATH=/tmp/elf-nonlinear13-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export MPLCONFIGDIR=/tmp/elf-nonlinear13-mpl-${SLURM_JOB_ID}
mkdir -p "$MIOPEN_USER_DB_PATH" "$MPLCONFIGDIR"
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest /lustre/orion/proj-shared/lrn105/yangc/envs/python312/bin/python -u scripts/train.py --out "runs/weekly13_lstm1_nonlinear_cosine_s42_${SLURM_JOB_ID}"

