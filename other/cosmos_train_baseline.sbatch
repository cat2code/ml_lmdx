#!/bin/bash
#SBATCH -J mldmx_baseline
#SBATCH -p gpua100i
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/hep/fs9/shared/ldmx/users/eliotmp/mpetren-msceng-ldmx}"
PROCESSED_CACHE_ROOT="${PROCESSED_CACHE_ROOT:-data/processed/production_5M_001_sharded}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cosmos_baselines}"
MODEL="${MODEL:-ECalTpadTransformer}"
EVENTS_PER_SOURCE="${EVENTS_PER_SOURCE:-500}"  # total events = 2 * this for 2e+3e
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-128}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-7}"
RUN_NAME="${RUN_NAME:-${MODEL}_${EVENTS_PER_SOURCE}per_source_${SLURM_JOB_ID:-manual}}"
RESUME="${RESUME:-}"

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

cd "${REPO_ROOT}/mldmx"
mkdir -p outputs/slurm "${OUTPUT_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
python -m pip install -e .

echo "host: $(hostname)"
echo "repo: ${REPO_ROOT}"
echo "model: ${MODEL}"
echo "processed cache root: ${PROCESSED_CACHE_ROOT}"
echo "events per source: ${EVENTS_PER_SOURCE}"
python --version
nvidia-smi
python -u scripts/check_cuda_environment.py

cmd=(
  python -u scripts/train_hit_classifier_baseline.py
  --model "${MODEL}"
  --processed-cache-root "${PROCESSED_CACHE_ROOT}"
  --events-per-source "${EVENTS_PER_SOURCE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --dim-feedforward "${DIM_FEEDFORWARD}"
  --dropout "${DROPOUT}"
  --seed "${SEED}"
  --device cuda
  --output-root "${OUTPUT_ROOT}"
  --run-name "${RUN_NAME}"
  --no-progress
)
if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

printf 'command:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
