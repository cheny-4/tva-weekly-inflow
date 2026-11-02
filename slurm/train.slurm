#!/bin/bash
#SBATCH --job-name=tva_weekly
#SBATCH --partition=service
#SBATCH --qos=develop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# Supply --account=YOUR_ACCOUNT when submitting from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f pipeline.py
: "${TVA_PYTHON:?Set TVA_PYTHON to an existing compatible PyTorch environment}"
module load rocm/7.1.1 craype-accel-amd-gfx90a
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export MIOPEN_USER_DB_PATH="/tmp/tva-weekly-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export MPLCONFIGDIR="/tmp/tva-weekly-mpl-${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MPLCONFIGDIR" outputs
TVA_MODEL="${TVA_MODEL:-total_runoff}"
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
  "$TVA_PYTHON" -u pipeline.py train --model "$TVA_MODEL" --out "outputs/train_${TVA_MODEL}_${SLURM_JOB_ID}"
