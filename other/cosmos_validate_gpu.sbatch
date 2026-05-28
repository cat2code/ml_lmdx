#!/bin/bash
#SBATCH -J mldmx_validate_gpu
#SBATCH -p gpua100i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/hep/fs9/shared/ldmx/users/eliotmp/mpetren-msceng-ldmx}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-data/processed/production_5M_001_sharded}"
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-10}"

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}/mldmx"
mkdir -p outputs/slurm
source "${REPO_ROOT}/.venv/bin/activate"
python -m pip install -e .

echo "host: $(hostname)"
echo "repo: ${REPO_ROOT}"
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
python --version
nvidia-smi
python -u scripts/check_cuda_environment.py
python -u scripts/validate_model_family_common_pipeline.py \
  --processed-cache-root "${PROCESSED_CACHE_ROOT}" \
  --events-per-source "${EVENTS_PER_SOURCE}" \
  --device cuda
