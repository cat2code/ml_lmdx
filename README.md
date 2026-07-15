# ml_ldmx

Machine-learning prototypes for LDMX event reconstruction, currently focused on
assigning ECal RecHits to incoming electrons using ECal information together
with TriggerPadTracks context. The newer pipelines are MLPF-inspired
transformer models that operate on variable-length event tokens.

![GNN Illustration](assets/data_pipeline.jpg)

## Current Focus

The active ECal/TriggerPad workflow reads ROOT events, builds per-event tensor
representations, trains hit-level prediction heads, and saves metrics,
checkpoints, and diagnostic plots.

Each context-aware event uses an 8-column node feature layout:

```text
[is_ecal, is_tpad, ecal_x, ecal_y, ecal_z, ecal_energy, tpad_centroid, tpad_pe]
```

`ecal_energy` is the reconstructed ECal RecHit energy input and `tpad_pe` is
the TriggerPadTracks photoelectron input. They are stored raw by default in the
Python CLIs; pass `--ecal-energy-transform log1p --tpad-pe-transform log1p`
during ROOT preprocessing or ROOT-backed cache creation to store
`log1p(max(value, 0))` instead. The truth deposited-energy fraction targets are
not log-transformed.

ECal nodes receive supervised targets; TriggerPadTracks nodes provide context.
The default target mode is `canonical-y`, which orders electron targets by
their spatial position in each event instead of relying on arbitrary physical
origin IDs.

Maintained training entry points are:

- `scripts/train_hit_classifier_baseline.py`: common canonical-y runner for
  the four maintained hit-origin classification baselines.
- `scripts/train_ecal_tpad_slot_model.py`: balanced `2e`/`3e` training with
  hit-origin, energy-fraction, slot-validity, and event electron-count heads.
- `scripts/train_ecal_tpad_mlpf_lite_scaled.py`: scalable three-origin
  hit/fraction training on a configurable ROOT subset, with reusable tensor
  caches.

## Setup

Run commands from this `ml_ldmx/` directory unless stated otherwise.

```powershell
cd ml_ldmx
python -m pip install -e .
```

`pyproject.toml` currently installs the local package only; it does not declare
runtime dependencies. The working environment must already provide the
scientific/ML stack used by the scripts, including PyTorch, PyTorch Geometric,
uproot, awkward, NumPy, and Matplotlib.

## Quick Start

Run the reusable smoke and unit tests from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The fast model-family smoke test first looks for processed events in
`data/processed/ecal_tpad_3class_smoke/`; if that directory is unavailable, set
`ML_LDMX_SMOKE_PROCESSED_DIR` to another tiny processed cache.

## Maintained Model Validation

Validate that all five maintained models consume the same canonical combined
event pipeline and canonical-y target convention:

```bash
python -m pytest tests/test_model_family_smoke.py -q
```

To validate against a sharded production-style cache, point the test at the
cache root and choose the device explicitly:

```bash
ML_LDMX_PROCESSED_CACHE_ROOT=data/processed/production_5M_001_sharded \
ML_LDMX_EVENTS_PER_SOURCE=10 \
ML_LDMX_ECAL_ENERGY_TRANSFORM=log1p \
ML_LDMX_TPAD_PE_TRANSFORM=log1p \
ML_LDMX_SMOKE_DEVICE=cpu \
python -m pytest tests/test_model_family_smoke.py -q
```

The model-family smoke runs forward/backward checks and writes no checkpoints
or plots. The two GravNet checks require the PyTorch Geometric `torch-cluster`
runtime dependency in the active environment.

### Slot Noise Experiment

The normal slot workflow filters noise hits at training time, so its
background output class receives no hit-level training examples by default.
To run the advanced model with explicit noise/background supervision:

```powershell
python scripts/train_ecal_tpad_slot_model.py `
  --supervise-noise `
  --events-per-class 100 `
  --epochs 20 `
  --device cpu `
  --run-name slot_noise_experiment
```

`--supervise-noise` is intentionally advanced-model-only. New sharded caches
retain aligned noise targets by default and can be used with this option;
legacy processed caches without `is_noise_target` cannot. Flagged noise hits
are assigned background class `0` and background-only fraction targets; they
do not enter canonical-y electron ordering or electron-count targets.
The slot trainer rejects the legacy `--keep-noise` option because it does not
supply background labels and can fail on flagged hits without contribution
truth; use `--supervise-noise` explicitly.

Run the focused optional ROOT-backed target/loss validation with:

```bash
ML_LDMX_ROOT_DATA=data/ldmx_overlay_events_700k python -m pytest tests/test_slot_model_noise_smoke.py -q
```

### Baseline Training

Maintained origin-classification experiments for the four baseline models use
one common runner and the same canonical-y event pipeline:

```powershell
python scripts/train_hit_classifier_baseline.py `
  --model ECalTpadTransformer `
  --events-per-class 100 `
  --epochs 20 `
  --device cuda `
  --run-name tpad_transformer_100_per_class
```

Valid `--model` values are `ECalGravNet`, `ECalTpadGravNet`,
`ECalTransformer`, and `ECalTpadTransformer`. The runner trains only ECal
hit-origin cross-entropy and writes classification histories, checkpoints,
test confusion matrices, and representative ECal truth/prediction plots.
For the ten-event processed smoke cache, pass:

```powershell
python scripts/train_hit_classifier_baseline.py `
  --model ECalTransformer `
  --processed-dir data/processed/ecal_tpad_3class_smoke `
  --allow-small-split `
  --epochs 1 `
  --device cpu
```

This runner supersedes the old `simple_3_class_classification_*.py`
prototypes for maintained baseline experiments. GravNet models require the
PyTorch Geometric `torch-cluster` runtime dependency.

### Inspecting a Saved Baseline Run

For the complete local 20,000-event 2e/3e Transformer showcase, including the
exact training, resume, inspection, and interactive-display commands, see
[`SUPERVISOR_DEMO.md`](SUPERVISOR_DEMO.md).

Re-run validation diagnostics and create interactive worst/median/best event
displays from a saved checkpoint without retraining:

```bash
python scripts/inspect_hit_classifier_run.py \
  --run-dir outputs/hit_classifier_baseline/my_run \
  --split val \
  --num-events 9 \
  --device auto
```

The inspector uses `checkpoints/best.pt` by default and writes to
`<run-dir>/inspection/best/<split>/`. Use `--checkpoint latest.pt` to select
another checkpoint. For a quick pass over a large split, add
`--max-inspection-events 1000`. If a run has moved between the cluster and a
local machine, point it at the relocated data with `--processed-dir`,
`--processed-cache`, `--processed-cache-root`, or repeated
`--processed-source` arguments. The inspection bundle includes the event
accuracy distribution, diagnostic-correlation panels, accuracy and
energy-weighted-accuracy profiles versus shower separation, and static plus
interactive worst/median/best event displays.

### Comparing Two Baseline Runs

After inspecting two models on the same saved split, build paired difficulty
profiles with:

```bash
python scripts/compare_hit_classifier_runs.py \
  --run transformer=outputs/hit_classifier_baseline/transformer_run \
  --run gravnet=outputs/hit_classifier_baseline/gravnet_run \
  --split val
```

The comparison requires identical event sets by default. It writes matched
per-event results, binned accuracy versus shower overlap and hit count,
confidence intervals, electron-count summaries, paired model differences, and
lists of events where both models fail or one model clearly outperforms the
other. The normalized overlap diagnostic is the minimum centroid separation
divided by the two showers' combined RMS width. By default, a hit belongs to
every shower with a positive truth contribution, with each membership counted
once and no fraction or energy weighting. Explicit `dominant_*` diagnostics
provide the alternative in which each hit belongs only to its largest-energy
origin. The primary definition uses all ECal layers, an `early_*` variant uses
the first three layers, and first-layer centroid distances are included in the
diagnostic-correlation plot.

Render event indices selected by the comparison with repeated arguments such
as `--event-index 123 --event-index 456` in
`scripts/inspect_hit_classifier_run.py`. Set `--num-events` at least as large
as the number requested when all selected displays should be written.

New processed caches preserve raw reconstructed ECal energy and raw
TriggerPadTracks `pe` separately from their optionally log-transformed model
inputs. This keeps energy-weighted diagnostics physically interpretable and
leaves the TriggerPad signal available for future diagnostics. Existing caches
remain supported and use their stored input-energy vector as a fallback.

### Assignment Ceilings and TPad Ablation

Analyze a saved best checkpoint without retraining:

```bash
python scripts/analyze_hit_classifier_ceiling.py \
  --run-dir outputs/hit_classifier_baseline/my_run \
  --checkpoint best.pt \
  --split val \
  --device auto
```

The command writes ordinary versus permutation-invariant accuracy, accuracy
versus truth deposited-energy fraction margin, TPad completeness summaries,
and a paired evaluation with every TPad token removed. It preserves the exact
saved event split and model preprocessing.

### Pipeline Benchmarking

Measure local IO, adapter, and representative forward/backward throughput
without writing run artifacts:

```powershell
python scripts/benchmark_common_pipeline.py `
  --processed-dir data/processed/ecal_tpad_3class_smoke `
  --root-dir data/ldmx_overlay_events_700k/3e/events `
  --max-events 5 `
  --device cpu
```

For smoke checks, use the processed smoke cache and small event limits. For
larger training jobs, prepare and reuse compatible processed canonical events
instead of repeatedly tensorizing ROOT input, and keep `--read-step-size`
available when ROOT preparation is required. Benchmark that chunk size on the
target filesystem; `50` is convenient for small local probes and the training
runners default to `500` for larger reads. When adapter overhead matters and
memory permits, pass `--cache-model-views` to the baseline trainer to derive
its selected model view once per loaded event and reuse it across epochs.
Without the optional `torch-cluster` dependency, the benchmark still reports
IO and transformer timings and marks GravNet model steps as skipped.

Train the balanced slot model on a small sample:

```powershell
python scripts/train_ecal_tpad_slot_model.py `
  --events-per-class 100 `
  --epochs 20 `
  --device cuda `
  --run-name slot_100_per_class
```

This script reads balanced samples from:

```text
data/ldmx_overlay_events_700k/2e/events/
data/ldmx_overlay_events_700k/3e/events/
```

It uses `data/processed/ecal_tpad_slot_model/` if processed tensor events have
already been created there. Results are written below
`outputs/ecal_tpad_slot_model/<run-name>/`.

Train the MLPF-lite transformer on a scalable `3e` subset:

```powershell
python scripts/train_ecal_tpad_mlpf_lite_scaled.py `
  --max-events 1000 `
  --epochs 20 `
  --device auto `
  --run-name mlpf_lite_1000
```

The scaled script reads `data/ldmx_overlay_events_700k/3e/events/` by default.
It automatically caches preprocessed events below
`data/processed/ecal_tpad_mlpf_lite_scaled/cache_<signature>/` and reuses the
cache when the ROOT inputs and preprocessing settings match. Add
`--force-preprocess` to rebuild a matching cache.

## Data Processing

For a single ROOT file, the preprocessing utility writes per-event `.pt`
tensors and a `manifest.json` file:

```powershell
python scripts/preprocess_ecal_tpad_dataset.py `
  --root-file path/to/events.root `
  --output-dir data/processed/my_dataset `
  --max-events 100
```

By default noise hits are filtered. Use `--keep-noise` to retain them and
`--no-edge-index` when only token tensors, rather than saved graph edges, are
needed.
Use `--ecal-energy-transform log1p --tpad-pe-transform log1p` here when you
want the saved tensors to use log-scaled reconstructed ECal energy and
TriggerPadTracks pe inputs.

### Scalable Sharded Cache

For large datasets, use ML-ready shards instead of the legacy
one-`.pt`-file-per-event format. Each source ROOT file becomes one processed
`.pt` shard, with `manifest.json` and `index.json` recording source files,
tensorization settings, event offsets, and shard contents:

```text
data/processed/ecal_tpad_2e3e_sharded/
  manifest.json
  index.json
  shards/
    shard_000001.pt
    shard_000002.pt
```

Create independent caches for the physical source datasets:

```powershell
python scripts/preprocess_ecal_tpad_sharded.py `
  --input-root-dir data/ldmx_overlay_events_700k/2e/events `
  --electron-count 2 `
  --source-label 2e `
  --output-dir data/processed/ecal_tpad_2e_sharded `
  --skip-existing

python scripts/preprocess_ecal_tpad_sharded.py `
  --input-root-dir data/ldmx_overlay_events_700k/3e/events `
  --electron-count 3 `
  --source-label 3e `
  --output-dir data/processed/ecal_tpad_3e_sharded `
  --skip-existing
```

ROOT files are ordered by numeric suffix (`events_2.root` precedes
`events_10.root`). Rerunning reuses valid completed shards and fills missing
ones; pass `--force` to rebuild. Sharded preprocessing retains explicit noise
targets by default. Use `--filter-noise` only to deliberately write a
noise-discarding cache. The reconstructed ECal energy and TriggerPadTracks pe
transforms are recorded in `manifest.json`; use
`--ecal-energy-transform log1p --tpad-pe-transform log1p` to create a log-input
cache, and pass the same options to training when the trainer creates or
validates caches. The Cosmos sbatch wrappers default both transforms to
`log1p`.

The current trainer interface can consume one sharded cache corresponding to
its configured training dataset directly, or a root directory containing
separate `2e/events` and `3e/events` caches:

```powershell
python scripts/train_hit_classifier_baseline.py `
  --model ECalTpadTransformer `
  --processed-cache-root data/processed/production_5M_001_sharded `
  --ecal-energy-transform log1p `
  --tpad-pe-transform log1p `
  --events-per-source 500 `
  --epochs 20 `
  --device cuda
```

With `--processed-cache-root`, the trainer lazily combines independent
`2e/events` and `3e/events` sharded caches without loading all shards into RAM.
`--events-per-source` is a balanced limit, so `500` means 1,000 total events
for the standard two-source layout. With `--processed-cache`, training reads a
single sharded cache lazily. In both modes `--events-per-class` is ignored; the
processed cache defines the available dataset. `--max-events` can restrict the
total consumed events, but `--events-per-source` is preferred for balanced
staged runs.

On a SLURM cluster, submit the generic preprocessing job once per independent
source dataset:

```bash
sbatch --export=ALL,SOURCE_LABEL=2e,ELECTRON_COUNT=2,ROOT_DIR=/path/to/2e/events,OUTPUT_DIR=/scratch/$USER/ml_ldmx/ecal_tpad_2e_sharded scripts/slurm/preprocess_ecal_tpad_sharded.sbatch
sbatch --export=ALL,SOURCE_LABEL=3e,ELECTRON_COUNT=3,ROOT_DIR=/path/to/3e/events,OUTPUT_DIR=/scratch/$USER/ml_ldmx/ecal_tpad_3e_sharded scripts/slurm/preprocess_ecal_tpad_sharded.sbatch
```

Quick ROOT-backed sharded smoke validation uses temporary shards and leaves no cache behind:

```bash
ML_LDMX_ROOT_DATA=data/ldmx_overlay_events_700k \
ML_LDMX_SHARD_SMOKE_MAX_ROOT_FILES=2 \
ML_LDMX_SHARD_SMOKE_MAX_EVENTS_PER_ROOT_FILE=10 \
python -m pytest tests/test_sharded_cache_smoke.py -q
```

The one-event-per-file processed format remains supported for existing small
datasets and prototype outputs.

## Cosmos GPU Training

Cosmos-specific launch notes live in `docs/cosmos_training.md`. The short
version is:

```bash
cd /path/to/ml_ldmx
git pull
source .venv/bin/activate
python -m pip install -e .
mkdir -p outputs/slurm
sbatch other/cosmos_validate_gpu.sbatch
sbatch --export=ALL,MODEL=ECalTpadTransformer,EVENTS_PER_SOURCE=500,EPOCHS=5,RUN_NAME=tpad_transformer_1k other/cosmos_train_baseline.sbatch
```

For the production cache layout, `--events-per-source 500` means 1,000 total
events across the separate `2e` and `3e` processed shard sources. Set
`SOURCE_LABEL=3e` or `SOURCE_LABEL=2e` to select only one processed source; in
that mode `EVENTS_PER_SOURCE` is the total number of events used. For example:

```bash
sbatch --export=ALL,MODEL=ECalTpadGravNet,SOURCE_LABEL=3e,EVENTS_PER_SOURCE=100000,EPOCHS=10,RUN_NAME=tpad_gravnet_3e_100k other/cosmos_train_baseline.sbatch
```

For a minimal ROOT-to-tensor inspection path, run the sharded preprocessor on one small source sample:

```bash
python scripts/preprocess_ecal_tpad_sharded.py \
  --input-root-dir path/to/3e/events \
  --electron-count 3 \
  --source-label 3e \
  --output-dir /tmp/ml_ldmx_one_file_shards \
  --ecal-energy-transform log1p \
  --tpad-pe-transform log1p \
  --max-root-files 1 \
  --max-events-per-root-file 10
```

## Training Outputs

Training runs create timestamped directories unless `--run-name` is supplied.
Common artifacts include:

- `config.json`, `history.json`, `history.csv`, and `train.log`
- `checkpoints/latest.pt`, `checkpoints/best.pt`, and periodic checkpoints
- `final_metrics.json`, loss/accuracy histories, and confusion matrices
- ECal truth/prediction plots and fraction diagnostics
- For the slot model, event-count predictions and event-count confusion plots

Resume a compatible run with `--resume path/to/checkpoints/latest.pt`. The
dataset split, label configuration, and target mode must match the checkpoint.

## Project Layout

```text
ml_ldmx/
  data/                         ROOT inputs and processed tensor caches
  outputs/                      Full training run artifacts
  figures/                      Prototype and notebook visualizations
  models/                       Saved weights from earlier simple experiments
  scripts/                      Runnable preprocessing, training, benchmark, and cluster entry points
  tests/                       Reusable unit and smoke regression tests
  src/ml_ldmx/
    io/                         ROOT reading, branch definitions, and artifact writers
    datasets/                   Tensorization, cached datasets, graph construction, preprocessing
    models/                     Transformer and graph neural-network architectures
    train/                      Losses, metrics, splits, checkpoints, and training loops
    eval/                       Validation and test evaluation for current pipelines
    viz/                        Training, ECal, fraction, and event-level plots
```

The older simple-classification prototype scripts have been removed from
`scripts/`; use the maintained baseline runner and the tests instead. The
slot-model and scaled MLPF-lite scripts contain the current end-to-end training
workflows.
