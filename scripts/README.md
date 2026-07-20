# ml_ldmx/scripts

This directory contains runnable Python and sbatch entry points for tasks the
user wants to perform: preprocessing, training, benchmarking, environment
checks, and cluster launch jobs.

Scripts should call reusable functionality from `ml_ldmx/src/ml_ldmx`; they
should not become the place where maintained features are implemented.

## Tensorize the 10M 2e and 10M 3e ROOT datasets on Slurm

The input directories are supplied at submission time. Each ROOT file becomes
one `.pt` shard. The launcher creates a fixed pool of `MAX_PARALLEL_JOBS` Slurm
workers; each worker processes several ROOT files sequentially.

```bash
cd /cluster/path/to/ml_ldmx
module load GCCcore/13.2.0 Python/3.11.5
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
mkdir -p outputs/slurm

sbatch \
  --export=ALL,DATASET_DIR=/cluster/data/production_5M_001,OUTPUT_ROOT=data/processed/production_10M_001_sharded,MAX_PARALLEL_JOBS=100 \
  scripts/sbatch/preprocess_production_5M_001_sharded.sbatch
```

`DATASET_DIR` must contain `2e/events/*.root` and `3e/events/*.root`. For any
other layout, replace it with `ROOT_2E_DIR=/path/to/2e` and
`ROOT_3E_DIR=/path/to/3e` in the `--export` list. Set
`MAX_PARALLEL_JOBS` to the largest concurrent job count allowed by the cluster.

```bash
squeue -u "$USER"
tail -F outputs/slurm/ml_ldmx_tensorize_JOB_ID.out
```

The dispatcher freezes the ordered ROOT-file plan, submits one fixed-size worker
array, and submits a dependent finalizer. For example, 3,003 ROOT files with
`MAX_PARALLEL_JOBS=100` use only 100 Slurm jobs, with about 30 files handled by
each worker. Unit and smoke tests are deliberately separate from this production
launcher. The finalizer requires exactly 10,000,000 events for each class
(20,000,000 total) and writes:

```text
data/processed/production_10M_001_sharded/2e/events/{manifest.json,index.json,shards/*.pt}
data/processed/production_10M_001_sharded/3e/events/{manifest.json,index.json,shards/*.pt}
data/processed/production_10M_001_sharded/preprocessing_summary.json
```

Both model inputs use `log1p`; every event also retains `ecal_raw_energy` and
`tpad_raw_pe`. Re-submit the same command to reuse completed, matching shards.
Set `EXPECTED_EVENTS_2E=0,EXPECTED_EVENTS_3E=0` only when intentionally building
a dataset whose size is not 10M per class.

## Train a maintained baseline on Cosmos

Tensorization produces ML-ready event shards; it does not train a model. The
baseline launcher reads those shards lazily and trains one of four maintained
ECal hit-origin classifiers with the same canonical-y targets and deterministic
data split:

- `ECalTransformer`: ECal hits only.
- `ECalTpadTransformer`: ECal hits with TPad context.
- `ECalGravNet`: ECal hits only, using GravNet layers.
- `ECalTpadGravNet`: ECal hits with TPad context and GravNet layers.

The normal comparison dataset contains balanced `2e` and `3e` sources. Noise
hits are filtered, continuous features are normalized using the training split
only, and the events are split deterministically into 80% train, 15%
validation, and 5% test using `SEED`.

### Check the training inputs and environment

Submit from the `ml_ldmx/` directory. The baseline batch script expects the
same environment prepared for tensorization and a completed cache with both
source directories:

```text
data/processed/production_10M_001_sharded/
  preprocessing_summary.json
  2e/events/
    manifest.json
    index.json
    shards/*.pt
  3e/events/
    manifest.json
    index.json
    shards/*.pt
```

Do not train from an unfinished cache. The manifests must describe the same
input transforms requested by the training job. The tensorization command
above writes `log1p` ECal energy and TPad PE inputs, which match the batch
script defaults.

Prepare the environment and Slurm log directory once:

```bash
cd /cluster/path/to/ml_ldmx
module load GCCcore/13.2.0 Python/3.11.5
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
mkdir -p outputs/slurm
```

`outputs/slurm` must exist before `sbatch`, because Slurm opens the log files
before the job body starts. The job itself reloads the Cosmos modules,
activates `${VENV_DIR:-<repo>/.venv}`, refreshes the editable install, prints a
CUDA report, and logs the exact Python training command.

### Start with a balanced smoke run

Use a small run to validate the full cache, GPU environment, loss calculation,
checkpointing, and plots before requesting a long job:

```bash
unset SOURCE_LABEL ELECTRON_COUNT RESUME
sbatch \
  --export=ALL,PROCESSED_CACHE_ROOT=data/processed/production_10M_001_sharded,MODEL=ECalTpadTransformer,EVENTS_PER_SOURCE=500,EPOCHS=5,BATCH_SIZE=8,SEED=7,OUTPUT_ROOT=outputs/cosmos_baselines,RUN_NAME=tpad_transformer_balanced_1k_seed7 \
  scripts/sbatch/cosmos_train_baseline.sbatch
```

With both sources selected, `EVENTS_PER_SOURCE=500` means 500 `2e` events plus
500 `3e` events, or 1,000 total. The deterministic split is therefore 800
training, 150 validation, and 50 test events. Leaving `SOURCE_LABEL` unset is
the recommended mode for maintained model comparisons. Because `--export=ALL`
forwards the submission shell, clear stale single-source or resume variables as
shown before every new balanced run.

The batch script defaults to `ECalTpadTransformer`, five epochs, batch size 8,
learning rate `1e-3`, no weight decay, and seed 7. The explicit values in the
example make the configuration traceable and the data split/training order
repeatable; GPU kernels are not guaranteed to be bitwise deterministic. The
directory name is also self-describing. Never reuse a `RUN_NAME` for an
unrelated experiment; files in an existing run directory can be replaced or
mixed. Reusing it is appropriate only when resuming that exact run.

Monitor the job with:

```bash
squeue -u "$USER"
tail -F outputs/slurm/ml_ldmx_baseline_JOB_ID.out
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

Before scaling, confirm in the Slurm log and run artifacts that:

- CUDA is available and the resolved training device is `cuda`.
- The resolved data directory is the intended production cache.
- The loaded event count and 80/15/5 split sizes are correct.
- Both input transforms are `log1p` for the cache produced above.
- Training and validation loss are finite, training loss decreases, and the
  predictions do not collapse to one class.
- `checkpoints/best.pt` and `checkpoints/latest.pt` were written.

There is no universal accuracy threshold for a successful smoke run. Its
purpose is to expose pipeline, memory, and learning failures cheaply.

### Scale in stages

`EVENTS_PER_SOURCE` is a per-source limit, not a total-event limit. For the
standard balanced cache:

| Purpose | `EVENTS_PER_SOURCE` | Total events | Starting epoch budget |
| --- | ---: | ---: | ---: |
| Pipeline smoke | 500 | 1,000 | 3–5 |
| Learning pilot | 5,000 | 10,000 | 5–10 |
| Model comparison | 50,000 | 100,000 | 10–20 |
| Serious candidate | 500,000 | 1,000,000 | chosen from pilot curves |
| Quarter-cache candidate | 2,500,000 | 5,000,000 | chosen from scaling results |
| Half-cache final | 5,000,000 | 10,000,000 | chosen from scaling results |
| Complete cache | 10,000,000 | 20,000,000 | chosen from scaling results |

The epoch ranges are starting points rather than fixed recommendations. Use
the loss curves and elapsed time from the preceding stage to choose the next
budget. Keep the dataset, event count, seed, batch size, optimizer settings,
and epoch budget fixed when comparing model families; change only `MODEL` and
use a distinct `RUN_NAME`.

For example, after the 1,000- and 10,000-event stages are healthy, submit a
100,000-event candidate run:

```bash
unset SOURCE_LABEL ELECTRON_COUNT RESUME
sbatch --time=24:00:00 --mem=64G \
  --export=ALL,PROCESSED_CACHE_ROOT=data/processed/production_10M_001_sharded,MODEL=ECalTpadTransformer,EVENTS_PER_SOURCE=50000,EPOCHS=15,BATCH_SIZE=8,SEED=7,OUTPUT_ROOT=outputs/cosmos_baselines,RUN_NAME=tpad_transformer_balanced_100k_seed7 \
  scripts/sbatch/cosmos_train_baseline.sbatch
```

Slurm resource overrides such as `--time` and `--mem` must appear before the
sbatch filename. Full-scale training and the final event-level diagnostic pass
can exceed the script's default 12-hour request, so size those resources from
measured pilot jobs rather than assuming the example is sufficient.

### Available batch settings

Override batch settings through `sbatch --export=ALL,NAME=value,...`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `REPO_ROOT` | Slurm submission directory | `ml_ldmx` repository directory |
| `VENV_DIR` | `<repo>/.venv` | Prepared Python virtual environment |
| `PROCESSED_CACHE_ROOT` | `data/processed/production_10M_001_sharded` | Root containing `2e/events` and `3e/events` |
| `MODEL` | `ECalTpadTransformer` | One of the four maintained baseline names |
| `EVENTS_PER_SOURCE` | `500` | Events selected from each active source |
| `SOURCE_LABEL` | empty | Balanced sources; set to `2e` or `3e` for one source |
| `ELECTRON_COUNT` | inferred | Electron count for a selected single source |
| `EPOCHS` | `5` | Total target epoch count |
| `BATCH_SIZE` | `8` | Events per optimizer step |
| `LR` | `1e-3` | AdamW learning rate |
| `WEIGHT_DECAY` | `0.0` | AdamW weight decay |
| `HIDDEN_DIM` | `64` | Hidden/model dimension |
| `NUM_LAYERS` | `2` | Transformer or GravNet layer count |
| `NUM_HEADS` | `4` | Transformer attention heads |
| `DIM_FEEDFORWARD` | `128` | Transformer feed-forward dimension |
| `DROPOUT` | `0.1` | Dropout probability |
| `SEED` | `7` | Data split and training-order seed |
| `ECAL_ENERGY_TRANSFORM` | `log1p` | Expected ECal transform recorded by the cache |
| `TPAD_PE_TRANSFORM` | `log1p` | Expected TPad transform recorded by the cache |
| `OUTPUT_ROOT` | `outputs/cosmos_baselines` | Parent directory for run artifacts |
| `RUN_NAME` | generated | Run directory name |
| `RESUME` | empty | Compatible `latest.pt` checkpoint to resume |

For Transformer models, `HIDDEN_DIM` must be divisible by `NUM_HEADS`. GravNet
models require the compiled PyTorch Geometric `torch-cluster` dependency;
`NUM_HEADS` and `DIM_FEEDFORWARD` do not affect them. Start with the defaults
and change one optimization or architecture choice at a time.

For a targeted single-source `3e` study:

```bash
sbatch \
  --export=ALL,PROCESSED_CACHE_ROOT=data/processed/production_10M_001_sharded,MODEL=ECalTpadGravNet,SOURCE_LABEL=3e,EVENTS_PER_SOURCE=100000,EPOCHS=10,SEED=7,RUN_NAME=tpad_gravnet_3e_100k_seed7 \
  scripts/sbatch/cosmos_train_baseline.sbatch
```

In single-source mode, `EVENTS_PER_SOURCE` is also the total event count and
the launcher reads only `<cache-root>/<SOURCE_LABEL>/events`. Balanced and
single-source results answer different questions and should not be placed in
the same model-comparison table. The wrapper uses the trainer's default valid
labels `1 2 3`; therefore a `2e`-only batch job still has a three-class output.
Use the Python training CLI directly with `--valid-labels 1 2` when a true
two-class `2e` experiment is intended.

### Training outputs and checkpoint choice

The smoke command above writes Slurm logs to:

```text
outputs/slurm/ml_ldmx_baseline_JOB_ID.out
outputs/slurm/ml_ldmx_baseline_JOB_ID.err
```

Training artifacts are written under
`<OUTPUT_ROOT>/<RUN_NAME>/`:

```text
config.json
run_overview.json
run_overview.md
model_architecture.txt
train.log
history.json
history.csv
loss_history.png
accuracy_history.png
checkpoints/
  best.pt
  latest.pt
  epoch_XXXX.pt
final_metrics.json
test_hit_origin_confusion_matrix.png
test_ecal_event_*_prediction_errors.png
val_event_accuracy.json
val_event_accuracy.csv
val_event_accuracy_overview.png
val_event_diagnostic_correlations.png
val_assignment_ceiling_diagnostics.png
val_shower_separation_profiles.png
val_representative_events.json
val_representative_events/*.{png,html}
```

Some diagnostic plots require enough finite events and the interactive HTML
requires Plotly, so small smoke runs may omit those files.

`run_overview.md` is the quickest human-readable reproducibility audit: it
records the model architecture, parameter count, resolved device, optimizer,
split sizes, class counts, preprocessing, and all training arguments.
`run_overview.json` contains the structured record, including the source-file
list.

`latest.pt` is the final completed epoch and is the checkpoint used to resume.
`best.pt` is selected by minimum validation loss and is the checkpoint to use
for model reporting and the analysis tutorial below. The trainer's top-level
validation diagnostics and `final_metrics.json` are calculated from the final
in-memory model, which may not be `best.pt`; regenerate validation and test
diagnostics from `best.pt` with `inspect_hit_classifier_run.py` before quoting
final results. Do not choose hyperparameters using the test split.

The trainer runs the complete `EPOCHS` budget; it does not stop early when
validation loss stops improving. The batch wrapper also leaves the trainer's
default per-epoch checkpointing enabled, so include periodic checkpoints in
the output-storage estimate for a long run.

### Resume an interrupted run

Resume from `latest.pt` with the same cache, source selection, event count,
model architecture, seed, transforms, optimizer settings, and `RUN_NAME`.
`EPOCHS` is the total target epoch count, not the number of additional epochs.
For example, to extend a ten-epoch run to twenty total epochs:

```bash
unset SOURCE_LABEL ELECTRON_COUNT
sbatch --time=24:00:00 --mem=64G \
  --export=ALL,PROCESSED_CACHE_ROOT=data/processed/production_10M_001_sharded,MODEL=ECalTpadTransformer,EVENTS_PER_SOURCE=5000,EPOCHS=20,BATCH_SIZE=8,SEED=7,OUTPUT_ROOT=outputs/cosmos_baselines,RUN_NAME=tpad_transformer_balanced_10k_seed7,RESUME=outputs/cosmos_baselines/tpad_transformer_balanced_10k_seed7/checkpoints/latest.pt \
  scripts/sbatch/cosmos_train_baseline.sbatch
```

Using the same run directory preserves the earlier `best.pt` if later epochs
do not improve validation loss. A new `RUN_NAME` is a new artifact bundle and
may have no `best.pt` unless the resumed model improves on the saved best loss.

Once training completes, continue with the best-checkpoint inspection,
ceiling, and comparison workflow below.

## Analyze saved hit-classifier runs

The three analysis entry points operate on runs produced by
`train_hit_classifier_baseline.py`:

- `inspect_hit_classifier_run.py` evaluates one saved checkpoint and creates
  event-level tables, summary plots, and representative event displays.
- `analyze_hit_classifier_ceiling.py` evaluates label-assignment ambiguity and
  measures the effect of removing every TPad token.
- `compare_hit_classifier_runs.py` compares two already generated event-record
  sets on matched events. It does not load a model or tensor shards.

They do not retrain or modify a checkpoint. The inspector supports the four
maintained baseline models (`ECalGravNet`, `ECalTpadGravNet`,
`ECalTransformer`, and `ECalTpadTransformer`). These scripts do not restore a
checkpoint from the separate slot-model trainer.

### What must exist before analysis

Run the commands below from the `ml_ldmx/` repository directory. The active
environment must contain the packages in `requirements.txt` and an editable
install of this repository:

```bash
cd /cluster/path/to/ml_ldmx
module load GCCcore/13.2.0 Python/3.11.5
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Keep the complete training run, especially this minimum layout:

```text
<run-dir>/
  config.json
  checkpoints/
    best.pt
    latest.pt
    epoch_XXXX.pt        # optional periodic checkpoints
```

The checkpoint stores the model state and constructor arguments, valid labels,
saved `train`/`val`/`test` index lists, and any fitted feature normalization.
`config.json` records the data source and total loaded event count. Retain both
when moving a run.

Inspection and ceiling analysis also need the exact processed events used for
training. The paths saved in the run are used when they still exist. A
production cache normally looks like:

```text
<processed-cache-root>/
  2e/events/{manifest.json,index.json,shards/*.pt}
  3e/events/{manifest.json,index.json,shards/*.pt}
```

If the data moved, use one—and only one—of these relocation forms:

```bash
# A balanced cache with both sources:
--processed-cache-root data/processed/production_10M_001_sharded

# A single-source run; repeat --processed-source only if the run used more sources:
--processed-source 2 2e data/processed/production_10M_001_sharded/2e/events
```

The relocated cache must contain the same events in the same source order and
use the same input transforms as training. The scripts check the total event
count and saved split indices, but event count alone cannot prove identical
ordering. The saved `--events-per-source` value is reused; do not change it
when relocating the cache. `--max-inspection-events` limits inference only; it
does not allow a smaller or differently ordered cache.

### 1. Inspect one checkpoint

This example evaluates the validation split from the validation-selected
checkpoint:

```bash
RUN_DIR=outputs/cosmos_baselines/my_run
CACHE_ROOT=data/processed/production_10M_001_sharded

python -u scripts/inspect_hit_classifier_run.py \
  --run-dir "$RUN_DIR" \
  --checkpoint best.pt \
  --split val \
  --processed-cache-root "$CACHE_ROOT" \
  --num-events 9 \
  --batch-size 32 \
  --device cuda
```

Omit the data override when the saved paths are still valid. If `--checkpoint`
is omitted, the inspector chooses `checkpoints/best.pt`, then falls back to
`checkpoints/latest.pt`. A named checkpoint such as `epoch_0010.pt` is also
accepted.

For a quick end-to-end check before evaluating a large split, use a separate
output directory so the subset does not overwrite or mix with a previous full
inspection:

```text
--max-inspection-events 1000 --num-events 3 \
--output-dir outputs/inspection_smoke/my_run_best_val
```

To inspect particular events, repeat `--event-index` and make `--num-events`
large enough to render all requested displays:

```bash
python -u scripts/inspect_hit_classifier_run.py \
  --run-dir "$RUN_DIR" \
  --checkpoint best.pt \
  --split val \
  --processed-cache-root "$CACHE_ROOT" \
  --event-index 1234 \
  --event-index 9876 \
  --num-events 2 \
  --device cuda \
  --output-dir outputs/manual_event_check
```

An event index is a global index in the loaded training dataset, not a position
within the split. It must belong to the saved split. Without `--output-dir`, the
inspector writes to:

```text
<run-dir>/inspection/<checkpoint-stem>/<split>/
```

For `best.pt` and `val`, the bundle is:

```text
inspection/best/val/
  inspection.log
  val_event_accuracy.json
  val_event_accuracy.csv
  val_event_accuracy_overview.png
  val_event_diagnostic_correlations.png
  val_assignment_ceiling_diagnostics.png
  val_shower_separation_profiles.png
  val_representative_events.json
  val_representative_events/
    *_prediction_errors.png
    *_interactive.html
  inspection_manifest.json
```

The JSON/CSV files contain one record per event, including accuracy, loss,
confidence, energy-weighted accuracy, shower-overlap diagnostics, TPad
completeness, and source provenance. The `generated_files` field in
`inspection_manifest.json` lists the analysis artifacts (excluding the log and
manifest itself). A plot can be omitted when too few finite diagnostic values
exist, and interactive HTML is skipped if Plotly is unavailable. The HTML
loads Plotly from a CDN when opened.

`hit_accuracy` in the manifest is hit-weighted across the split. In contrast,
`mean_event_accuracy` gives every event equal weight. Smaller normalized shower
separation means more overlap between showers and usually a harder event.

### 2. Measure assignment ambiguity and TPad reliance

This analysis makes two paired inference passes: the normal event and the same
event with every TPad token removed.

```bash
python -u scripts/analyze_hit_classifier_ceiling.py \
  --run-dir "$RUN_DIR" \
  --checkpoint best.pt \
  --split val \
  --processed-cache-root "$CACHE_ROOT" \
  --batch-size 32 \
  --device cuda
```

This script requires a combined ECal+TPad checkpoint:
`ECalTpadTransformer` or `ECalTpadGravNet`. ECal-only checkpoints have no TPad
tokens to remove and are rejected. Expect roughly twice the inference work of
a metrics-only inspection. Use `--max-inspection-events` for a small trial.
Give that trial a separate `--output-dir`, such as
`outputs/ceiling_smoke/my_run_best_val`, to avoid replacing full-split results.

The default output directory is
`<run-dir>/ceiling_analysis/<checkpoint-stem>/<split>/` and contains:

```text
ceiling_analysis/best/val/
  ceiling_analysis.log
  reference_event_accuracy.json
  reference_event_accuracy.csv
  tpad_ablated_event_accuracy.json
  tpad_ablated_event_accuracy.csv
  reference_assignment_ceiling_diagnostics.png
  tpad_ablation.png
  ceiling_summary.json
  ceiling_manifest.json
```

In `ceiling_summary.json`, a positive `tpad_hit_accuracy_gain` means the intact
model was more accurate than the TPad-ablated evaluation; a negative value
means removal improved accuracy. Permutation-invariant accuracy is an oracle
diagnostic after choosing the best single global relabeling for each event. A
large ordinary-to-permutation-invariant gain suggests event-wide label binding
or label swapping; it is not a separately trained score. A small top-two truth
deposited-energy-fraction margin marks intrinsically mixed hit targets.

### 3. Compare two inspected checkpoints

First inspect both checkpoints on the same split, complete event set, and
diagnostic radius. Then pass the exact inspection directories or record files:

```bash
INSPECTION_A=outputs/cosmos_baselines/ecal_transformer/inspection/best/val
INSPECTION_B=outputs/cosmos_baselines/ecal_tpad_transformer/inspection/best/val

python -u scripts/compare_hit_classifier_runs.py \
  --run "ECal-only=$INSPECTION_A" \
  --run "ECal+TPad=$INSPECTION_B" \
  --split val \
  --output-dir outputs/hit_classifier_comparisons/ecal_vs_tpad_val
```

Exactly two distinct `LABEL=PATH` values are required. Labels are display
names; the first run minus the second run determines the sign of every paired
accuracy delta. `PATH` may be an inspection directory, an exact event-accuracy
JSON/CSV file, or a run directory.

Prefer an inspection directory or exact file for checkpoint comparisons. When
given a run directory, the resolver checks a top-level
`<split>_event_accuracy.json` before
`inspection/<checkpoint>/<split>/...`; the top-level file can describe the
trainer's final in-memory model rather than `best.pt`. Also note that
`--checkpoint best` in the comparison CLI names an inspection directory and
therefore has no `.pt` suffix.

By default, both record sets must contain identical event identities and
matching geometry. `--allow-partial-match` deliberately compares only their
intersection. The comparison creates:

```text
comparison.log
matched_event_comparison.csv
binned_difficulty_profiles.csv
accuracy_by_electron_count.csv
comparison_summary.json
interesting_events.json
accuracy_difficulty_profiles.png
energy_weighted_accuracy_difficulty_profiles.png
paired_event_accuracy.png
accuracy_by_electron_count.png
comparison_manifest.json
```

The summary includes hit-weighted and mean-event metrics, paired differences,
95% confidence intervals, resolved inputs, and matching counts. The interesting
event file identifies events where both models fail, both perform well, or one
outperforms the other. Some profile CSVs/plots are omitted when their required
fields are absent or constant. For up to 20,000 matched events, use
`--bootstrap-samples 0` to replace resampling with a faster normal
approximation. The script selects the normal approximation automatically above
20,000 observations.

### Run the analyses on Cosmos with Slurm

Create the log directory before submission; Slurm opens the log files before
the job body can create it:

```bash
cd /cluster/path/to/ml_ldmx
mkdir -p outputs/slurm
```

`cosmos_hit_classifier_analysis.sbatch` takes a mode followed by the unchanged
Python CLI arguments. Submit an inspection with:

```bash
sbatch scripts/sbatch/cosmos_hit_classifier_analysis.sbatch inspect \
  --run-dir outputs/cosmos_baselines/my_run \
  --checkpoint best.pt \
  --split val \
  --processed-cache-root data/processed/production_10M_001_sharded \
  --num-events 9 \
  --batch-size 32 \
  --device cuda
```

Submit the TPad ablation analysis with:

```bash
sbatch scripts/sbatch/cosmos_hit_classifier_analysis.sbatch ceiling \
  --run-dir outputs/cosmos_baselines/my_tpad_run \
  --checkpoint best.pt \
  --split val \
  --processed-cache-root data/processed/production_10M_001_sharded \
  --batch-size 32 \
  --device cuda
```

After both inspection jobs finish successfully, submit the comparison using
their exact output directories:

```bash
sbatch scripts/sbatch/cosmos_hit_classifier_analysis.sbatch compare \
  --run "ECal-only=outputs/cosmos_baselines/ecal_transformer/inspection/best/val" \
  --run "ECal+TPad=outputs/cosmos_baselines/ecal_tpad_transformer/inspection/best/val" \
  --split val \
  --output-dir outputs/hit_classifier_comparisons/ecal_vs_tpad_val
```

The wrapper uses the established Cosmos modules (`GCCcore/13.2.0` and
`Python/3.11.5`), finds `.venv` in the repository or its parent, requests one
A100 GPU, and writes `outputs/slurm/ml_ldmx_analysis_JOB_ID.{out,err}`. Set
`VENV_DIR`, `REPO_ROOT`, `CLUSTER_MODULES`, or `SKIP_MODULE_LOAD=1` in the
submission environment when the defaults do not apply. Slurm resource options
such as `--time` and `--mem` must appear before the sbatch filename.

The default 32 GiB and 12-hour request is a starting point, not a full-scale
guarantee. Inspection retains all selected event records in memory; ceiling
analysis retains both reference and ablated records and performs two passes.
Start with `--max-inspection-events`, inspect the Slurm memory/runtime report,
then increase resources for a large split, for example:

```bash
sbatch --mem=64G --time=24:00:00 \
  scripts/sbatch/cosmos_hit_classifier_analysis.sbatch ceiling \
  --run-dir outputs/cosmos_baselines/my_tpad_run \
  --checkpoint best.pt \
  --split val \
  --device cuda
```

Inspection and ceiling inference benefit from the requested GPU. Comparison is
CPU-only once its JSON/CSV inputs exist, although this common wrapper retains
the same known Cosmos GPU request. Monitor jobs and logs with:

```bash
squeue -u "$USER"
tail -F outputs/slurm/ml_ldmx_analysis_JOB_ID.out
```

For a complete worked training and inspection example, see
[`../SUPERVISOR_DEMO.md`](../SUPERVISOR_DEMO.md).

## Test scripts belong in ml_ldmx/tests

Reusable checks for behavior that should remain stable between updates belong in
`ml_ldmx/tests`, preferably as `test_*.py` files that can be run together with:

```bash
python -m unittest discover -s tests
```

A script under `scripts/` may still create a small smoke dataset or launch a
cluster preflight job, but regression-style assertions should live in `tests/`.
