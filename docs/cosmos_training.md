# Cosmos Training Guide

This guide is for running the maintained MLDMX workflow on Cosmos with
already tensorized ML-ready shards. The normal training path should not read
ROOT files: it should load processed shards from
`data/processed/production_5M_001_sharded`.

## Setup

Log in, enter the repository, pull updates, and refresh editable install:

```bash
ssh eliotmp@cosmos.lunarc.lu.se
cd /projects/hep/fs9/shared/ldmx/users/eliotmp/mpetren-msceng-ldmx
git pull
source .venv/bin/activate
cd mldmx
python -m pip install -e .
mkdir -p outputs/slurm
```

The Slurm scripts load the same modules as the working CUDA test:
`GCCcore/13.2.0` and `Python/3.11.5`. If the module stack changes, run
`module spider CUDA` and update the batch scripts after a small CUDA check.

Expected production shard layout:

```text
data/processed/production_5M_001_sharded/
  2e/events/
    manifest.json
    index.json
    shards/
  3e/events/
    manifest.json
    index.json
    shards/
```

## CUDA Check

Interactive/login-node-safe check:

```bash
python scripts/check_cuda_environment.py
```

GPU batch validation:

```bash
sbatch other/cosmos_validate_gpu.sbatch
```

This runs the CUDA check and the five-model common-pipeline validation on GPU.

## Event Counts

For the production layout, use `--processed-cache-root` and
`--events-per-source`. With the standard `2e` plus `3e` sources:

```text
total events = 2 * events_per_source
```

Use balanced staged counts:

```text
total events     --events-per-source
1,000            500
10,000           5,000
100,000          50,000
1,000,000        500,000
5,000,000        2,500,000
```

`--max-events` is a total cap, but it truncates the combined source order.
Prefer `--events-per-source` for balanced `2e`/`3e` comparisons.

## First Training Runs

Run a 1,000-event baseline job:

```bash
sbatch --export=ALL,MODEL=ECalTpadTransformer,EVENTS_PER_SOURCE=500,EPOCHS=5,RUN_NAME=tpad_transformer_1k \
  other/cosmos_train_baseline.sbatch
```

Run a 10,000-event baseline job:

```bash
sbatch --export=ALL,MODEL=ECalTpadTransformer,EVENTS_PER_SOURCE=5000,EPOCHS=10,RUN_NAME=tpad_transformer_10k \
  other/cosmos_train_baseline.sbatch
```

Run a 1,000-event advanced slot-model job:

```bash
sbatch --export=ALL,EVENTS_PER_SOURCE=500,EPOCHS=5,RUN_NAME=slot_1k \
  other/cosmos_train_slot.sbatch
```

Run the slot model with explicit noise/background supervision, only when the
processed shards contain `is_noise_target`:

```bash
sbatch --export=ALL,EVENTS_PER_SOURCE=500,EPOCHS=5,SUPERVISE_NOISE=1,RUN_NAME=slot_noise_1k \
  other/cosmos_train_slot.sbatch
```

## Hyperparameters

The batch scripts expose the main choices as environment variables:

```text
MODEL                 baseline model name
PROCESSED_CACHE_ROOT  production shard root
EVENTS_PER_SOURCE     balanced events per source
EPOCHS                number of epochs
BATCH_SIZE            events per optimizer step
LR                    learning rate
WEIGHT_DECAY          optimizer weight decay
HIDDEN_DIM            model hidden dimension
NUM_LAYERS            transformer/slot layers
NUM_HEADS             attention heads
DIM_FEEDFORWARD       transformer feed-forward size
DROPOUT               dropout
OUTPUT_ROOT           output directory root
RUN_NAME              output run directory name
RESUME                checkpoint path to resume
```

Override any of them with `sbatch --export=ALL,NAME=value,...`.

## Outputs

Slurm logs:

```text
outputs/slurm/<job-name>_<job-id>.out
outputs/slurm/<job-name>_<job-id>.err
```

Training artifacts:

```text
outputs/cosmos_baselines/<run-name>/
outputs/cosmos_slot/<run-name>/
```

Each run writes `config.json`, `history.json`, `history.csv`, `train.log`,
`final_metrics.json`, checkpoints in `checkpoints/`, confusion matrices, and
representative ECal truth/prediction plots. The slot trainer also writes
multi-task diagnostics such as fraction and count metrics.

## Monitoring

```bash
jobinfo -u $USER
squeue -j <jobid>
tail -F outputs/slurm/<job-name>_<jobid>.out
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
srun --jobid=<jobid> --pty nvidia-smi
```

## Resume

Resume from the last checkpoint:

```bash
sbatch --export=ALL,MODEL=ECalTpadTransformer,EVENTS_PER_SOURCE=5000,RUN_NAME=tpad_transformer_10k_resume,RESUME=outputs/cosmos_baselines/tpad_transformer_10k/checkpoints/latest.pt \
  other/cosmos_train_baseline.sbatch
```

For the slot model, use `other/cosmos_train_slot.sbatch` and a checkpoint from
`outputs/cosmos_slot/<run-name>/checkpoints/latest.pt`.

## Recommended Scaling Plan

Stage 0: CUDA environment check.

Stage 1: five-model validation on GPU.

Stage 2: 1,000-event training sanity run.

Stage 3: 10,000-event learning-curve run.

Stage 4: 100,000-event hyperparameter comparison.

Stage 5: 1,000,000-event serious candidate training.

Stage 6: up to 5,000,000-event final-scale training.

At every stage, inspect loss curves, validation metrics, confusion matrices,
and prediction plots before scaling. Do not spend A100 time on bigger runs
until the smaller run behaves scientifically and technically.

## Remaining Cluster-Specific Checks

The exact partition/account/QOS limits may differ by allocation. The default
scripts use `gpua100i` with one GPU. Use `gpua100` only when an exclusive
A100 80 GB job is actually needed. Do not use `gpua40` for batch training.

GravNet models require PyTorch Geometric's compiled dependencies, especially
`torch-cluster`; installing `torch-scatter` is also recommended for faster
scatter operations.
