#!/usr/bin/env bash
#SBATCH --job-name=ml_ldmx_tensorize
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

MODE="${MODE:-dispatch}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/data/processed/production_10M_001_sharded}"
PLAN_FILE="${PLAN_FILE:-}"
TASK_INDEX_OFFSET="${TASK_INDEX_OFFSET:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
READ_STEP_SIZE="${READ_STEP_SIZE:-500}"
ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM:-log1p}"
TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM:-log1p}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-100}"
MAX_ARRAY_SIZE="${MAX_ARRAY_SIZE:-}"
MAX_ARRAY_SIZE_FALLBACK="${MAX_ARRAY_SIZE_FALLBACK:-1000}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-1}"
MEM_PER_WORKER="${MEM_PER_WORKER:-32G}"
WORKER_TIME="${WORKER_TIME:-24:00:00}"
FINALIZER_MEM="${FINALIZER_MEM:-16G}"
FINALIZER_TIME="${FINALIZER_TIME:-04:00:00}"
EXPECTED_EVENTS_2E="${EXPECTED_EVENTS_2E:-10000000}"
EXPECTED_EVENTS_3E="${EXPECTED_EVENTS_3E:-10000000}"
RUN_TESTS="${RUN_TESTS:-1}"
FULL_VALIDATE="${FULL_VALIDATE:-0}"
ALLOW_FAILED_ROOT_FILES="${ALLOW_FAILED_ROOT_FILES:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MODULE_LOAD="${SKIP_MODULE_LOAD:-0}"
CLUSTER_MODULES="${CLUSTER_MODULES:-GCCcore/13.2.0 Python/3.11.5}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

setup_environment() {
  [[ -f "${REPO_DIR}/scripts/preprocess_ecal_tpad_array.py" ]] || \
    fail "REPO_DIR does not point to ml_ldmx: ${REPO_DIR}"

  if [[ "${SKIP_MODULE_LOAD}" != "1" ]]; then
    type module >/dev/null 2>&1 || \
      fail "The module command is unavailable. Set SKIP_MODULE_LOAD=1 or configure CLUSTER_MODULES."
    module purge
    # Module names are intentionally configurable for clusters other than Cosmos.
    module load ${CLUSTER_MODULES}
  fi

  if [[ -z "${VENV_DIR:-}" ]]; then
    if [[ -f "${REPO_DIR}/.venv/bin/activate" ]]; then
      VENV_DIR="${REPO_DIR}/.venv"
    elif [[ -f "${REPO_DIR}/../.venv/bin/activate" ]]; then
      VENV_DIR="$(cd "${REPO_DIR}/../.venv" && pwd)"
    else
      fail "No virtualenv found. Set VENV_DIR or create ${REPO_DIR}/.venv."
    fi
  fi
  [[ -f "${VENV_DIR}/bin/activate" ]] || fail "Missing virtualenv: ${VENV_DIR}"

  cd "${REPO_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"
  "${PYTHON_BIN}" -c "import awkward, ml_ldmx, torch, uproot; print('Python dependencies are available')"
}

dispatch() {
  [[ -n "${OUTPUT_ROOT}" ]] || fail "Set OUTPUT_ROOT to the destination directory on shared storage."

  if [[ -n "${DATASET_DIR:-}" ]]; then
    ROOT_2E_DIR="${ROOT_2E_DIR:-${DATASET_DIR}/2e/events}"
    ROOT_3E_DIR="${ROOT_3E_DIR:-${DATASET_DIR}/3e/events}"
  fi
  [[ -n "${ROOT_2E_DIR:-}" ]] || fail "Set ROOT_2E_DIR, or set DATASET_DIR with a 2e/events directory."
  [[ -n "${ROOT_3E_DIR:-}" ]] || fail "Set ROOT_3E_DIR, or set DATASET_DIR with a 3e/events directory."
  [[ -d "${ROOT_2E_DIR}" ]] || fail "Missing 2e ROOT directory: ${ROOT_2E_DIR}"
  [[ -d "${ROOT_3E_DIR}" ]] || fail "Missing 3e ROOT directory: ${ROOT_3E_DIR}"
  [[ "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_PARALLEL_JOBS must be positive."

  setup_environment
  mkdir -p "${OUTPUT_ROOT}" "${REPO_DIR}/outputs/slurm" "${OUTPUT_ROOT}/_slurm"
  local run_id="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
  PLAN_FILE="${OUTPUT_ROOT}/_slurm/plan_${run_id}.json"

  echo "Repository:       ${REPO_DIR}"
  echo "2e ROOT input:    ${ROOT_2E_DIR}"
  echo "3e ROOT input:    ${ROOT_3E_DIR}"
  echo "Shard output:     ${OUTPUT_ROOT}"
  echo "Worker limit:     ${MAX_PARALLEL_JOBS}"
  echo "ECal transform:   ${ECAL_ENERGY_TRANSFORM}"
  echo "TPad transform:   ${TPAD_PE_TRANSFORM}"
  echo "Expected events:  2e=${EXPECTED_EVENTS_2E}, 3e=${EXPECTED_EVENTS_3E}"

  if [[ "${RUN_TESTS}" == "1" ]]; then
    echo "Running the Python test suite..."
    "${PYTHON_BIN}" -m unittest discover -s tests
    echo "Running the ROOT-backed tensorization/reader/training smoke test..."
    ML_LDMX_ROOT_2E_DIR="${ROOT_2E_DIR}" \
      ML_LDMX_ROOT_3E_DIR="${ROOT_3E_DIR}" \
      ML_LDMX_ECAL_ENERGY_TRANSFORM="${ECAL_ENERGY_TRANSFORM}" \
      ML_LDMX_TPAD_PE_TRANSFORM="${TPAD_PE_TRANSFORM}" \
      "${PYTHON_BIN}" -m unittest discover -s tests -p test_sharded_cache_smoke.py
  fi

  "${PYTHON_BIN}" -u scripts/preprocess_ecal_tpad_array.py plan \
    --plan-file "${PLAN_FILE}" \
    --output-root "${OUTPUT_ROOT}" \
    --source 2 2e "${ROOT_2E_DIR}" \
    --source 3 3e "${ROOT_3E_DIR}" \
    --ecal-energy-transform "${ECAL_ENERGY_TRANSFORM}" \
    --tpad-pe-transform "${TPAD_PE_TRANSFORM}"

  local task_count max_array_size array_count
  task_count=$("${PYTHON_BIN}" scripts/preprocess_ecal_tpad_array.py count --plan-file "${PLAN_FILE}")
  [[ "${task_count}" =~ ^[1-9][0-9]*$ ]] || fail "The plan contains no worker tasks."
  echo "Frozen plan:      ${PLAN_FILE}"
  echo "Array tasks:      ${task_count}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: plan and tests succeeded; no Slurm workers were submitted."
    return
  fi

  if [[ -n "${MAX_ARRAY_SIZE}" ]]; then
    [[ "${MAX_ARRAY_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_ARRAY_SIZE must be a positive integer."
    max_array_size="${MAX_ARRAY_SIZE}"
  else
    if ! max_array_size=$(scontrol show config 2>/dev/null | awk '$1 == "MaxArraySize" {print $3; exit}'); then
      max_array_size=""
    fi
    if [[ ! "${max_array_size}" =~ ^[1-9][0-9]*$ ]]; then
      [[ "${MAX_ARRAY_SIZE_FALLBACK}" =~ ^[1-9][0-9]*$ ]] || \
        fail "MAX_ARRAY_SIZE_FALLBACK must be a positive integer."
      max_array_size="${MAX_ARRAY_SIZE_FALLBACK}"
      echo "WARNING: Slurm MaxArraySize is unavailable; using safe fallback ${max_array_size}."
    fi
  fi
  array_count=$(((task_count + max_array_size - 1) / max_array_size))
  echo "Slurm array limit: ${max_array_size} task(s) per array"
  echo "Worker arrays:     ${array_count} sequential chunk(s)"

  local worker_submission worker_job_id finalizer_submission finalizer_job_id
  local previous_worker_job_id=""
  local task_offset=0
  local chunk_number=1
  local chunk_task_count chunk_last_task task_stop
  local -a worker_sbatch_args worker_job_ids
  worker_job_ids=()

  while ((task_offset < task_count)); do
    chunk_task_count=$((task_count - task_offset))
    if ((chunk_task_count > max_array_size)); then
      chunk_task_count="${max_array_size}"
    fi
    chunk_last_task=$((chunk_task_count - 1))
    task_stop=$((task_offset + chunk_task_count - 1))
    worker_sbatch_args=(
      --parsable
      --export=ALL
      --job-name=ml_ldmx_tensor_worker
    )
    if [[ -n "${previous_worker_job_id}" ]]; then
      worker_sbatch_args+=(--dependency="afterany:${previous_worker_job_id}")
    fi
    worker_sbatch_args+=(
      --array="0-${chunk_last_task}%${MAX_PARALLEL_JOBS}"
      --cpus-per-task="${CPUS_PER_WORKER}"
      --mem="${MEM_PER_WORKER}"
      --time="${WORKER_TIME}"
      --output="${REPO_DIR}/outputs/slurm/%x_%A_%a.out"
      --error="${REPO_DIR}/outputs/slurm/%x_%A_%a.err"
    )

    worker_submission=$(
      MODE=worker PLAN_FILE="${PLAN_FILE}" REPO_DIR="${REPO_DIR}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
        TASK_INDEX_OFFSET="${task_offset}" \
        sbatch "${worker_sbatch_args[@]}" \
          "${REPO_DIR}/scripts/sbatch/preprocess_production_5M_001_sharded.sbatch"
    )
    worker_job_id="${worker_submission%%;*}"
    worker_job_ids+=("${worker_job_id}")
    echo "Worker array job ${chunk_number}/${array_count}: ${worker_job_id} (plan tasks ${task_offset}-${task_stop})"

    previous_worker_job_id="${worker_job_id}"
    task_offset=$((task_stop + 1))
    chunk_number=$((chunk_number + 1))
  done

  finalizer_submission=$(
    MODE=finalize PLAN_FILE="${PLAN_FILE}" REPO_DIR="${REPO_DIR}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
      sbatch --parsable --export=ALL \
      --job-name=ml_ldmx_tensor_finalize \
      --dependency="afterany:${previous_worker_job_id}" \
      --cpus-per-task=1 \
      --mem="${FINALIZER_MEM}" \
      --time="${FINALIZER_TIME}" \
      --output="${REPO_DIR}/outputs/slurm/%x_%j.out" \
      --error="${REPO_DIR}/outputs/slurm/%x_%j.err" \
      "${REPO_DIR}/scripts/sbatch/preprocess_production_5M_001_sharded.sbatch"
  )
  finalizer_job_id="${finalizer_submission%%;*}"

  local worker_job_list
  worker_job_list=$(IFS=,; echo "${worker_job_ids[*]}")
  echo "Worker array jobs: ${worker_job_list}"
  echo "Finalizer job:    ${finalizer_job_id}"
  echo "Track all:        squeue -j ${worker_job_list},${finalizer_job_id}"
}

run_worker() {
  [[ -n "${PLAN_FILE}" ]] || fail "Worker received no PLAN_FILE."
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "Worker received no SLURM_ARRAY_TASK_ID."
  [[ "${TASK_INDEX_OFFSET}" =~ ^[0-9]+$ ]] || fail "TASK_INDEX_OFFSET must be a non-negative integer."
  setup_environment
  local task_index=$((TASK_INDEX_OFFSET + SLURM_ARRAY_TASK_ID))
  local worker_args=(
    scripts/preprocess_ecal_tpad_array.py worker
    --plan-file "${PLAN_FILE}"
    --task-index "${task_index}"
    --read-step-size "${READ_STEP_SIZE}"
  )
  [[ "${FORCE}" == "1" ]] && worker_args+=(--force)
  "${PYTHON_BIN}" -u "${worker_args[@]}"
}

finalize() {
  [[ -n "${PLAN_FILE}" ]] || fail "Finalizer received no PLAN_FILE."
  setup_environment
  local finalize_args=(
    scripts/preprocess_ecal_tpad_array.py finalize
    --plan-file "${PLAN_FILE}"
    --expected-events-2e "${EXPECTED_EVENTS_2E}"
    --expected-events-3e "${EXPECTED_EVENTS_3E}"
  )
  [[ "${ALLOW_FAILED_ROOT_FILES}" == "1" ]] && finalize_args+=(--allow-failed-root-files)
  [[ "${FULL_VALIDATE}" == "1" ]] && finalize_args+=(--load-all-shards)
  "${PYTHON_BIN}" -u "${finalize_args[@]}"
  echo "ML-ready caches: ${OUTPUT_ROOT}/{2e,3e}/events"
  echo "Summary:         ${OUTPUT_ROOT}/preprocessing_summary.json"
}

case "${MODE}" in
  dispatch) dispatch ;;
  worker) run_worker ;;
  finalize) finalize ;;
  *) fail "Unknown MODE: ${MODE}" ;;
esac
