# Transformer Supervisor Demo

This guide reproduces the local showcase using separate
`ECalTpadTransformer` hit-origin classifiers for 2-electron and 3-electron
overlay events. Each run uses 20,000 tensorized events, seed 7, and five
epochs. The deterministic split is 16,000 training, 3,000 validation, and
1,000 test events.

The demo is large enough to expose event-to-event performance variation and
shower-overlap trends. It is not a final hyperparameter study.

## Current Demo Results

Both reported models are `checkpoints/best.pt`, selected only by validation
loss. In both runs this is epoch 4.

| sample | split | hit accuracy | mean event accuracy | median event accuracy | mean event energy-weighted accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| 2e | validation | 0.8223 | 0.8167 | 0.8511 | 0.8621 |
| 2e | test | 0.8203 | 0.8142 | 0.8513 | 0.8589 |
| 3e | validation | 0.7270 | 0.7219 | 0.7469 | 0.7807 |
| 3e | test | 0.7277 | 0.7226 | 0.7533 | 0.7818 |

Training took about 15 minutes for 2e and 19 minutes for 3e on the Mac used
for this demo. Runtime depends strongly on the machine and PyTorch version.

The generated result index is
[`outputs/supervisor_demo_transformer_20k/DEMO_INDEX.md`](outputs/supervisor_demo_transformer_20k/DEMO_INDEX.md).

## 1. Environment

Run every command from `ml_ldmx/`:

```bash
cd /Users/eliotmontesinopetren/src/mpetren-msceng-ldmx/ml_ldmx
source ../.venv/bin/activate
```

Check PyTorch and the available accelerator:

```bash
python -c "import torch; print('torch', torch.__version__); print('MPS', torch.backends.mps.is_available()); print('CUDA', torch.cuda.is_available())"
```

The expected input directories are:

```text
data/ldmx_overlay_events_700k_shards_log1p/2e/events
data/ldmx_overlay_events_700k_shards_log1p/3e/events
```

Their manifests must report both `ecal_energy_transform: log1p` and
`tpad_pe_transform: log1p`. The training commands below explicitly require
those cache settings. Raw reconstructed ECal energy remains available for the
energy-weighted diagnostics.

## 2. Train the 2e Model

```bash
python scripts/train_hit_classifier_baseline.py \
  --model ECalTpadTransformer \
  --processed-source 2 2e data/ldmx_overlay_events_700k_shards_log1p/2e/events \
  --events-per-source 20000 \
  --valid-labels 1 2 \
  --ecal-energy-transform log1p \
  --tpad-pe-transform log1p \
  --epochs 5 \
  --batch-size 8 \
  --seed 7 \
  --device mps \
  --cache-model-views \
  --no-progress \
  --num-ecal-plots 2 \
  --num-diagnostic-event-displays 3 \
  --output-root outputs/supervisor_demo_transformer_20k \
  --run-name transformer_2e_20k_seed7
```

## 3. Train the 3e Model

```bash
python scripts/train_hit_classifier_baseline.py \
  --model ECalTpadTransformer \
  --processed-source 3 3e data/ldmx_overlay_events_700k_shards_log1p/3e/events \
  --events-per-source 20000 \
  --valid-labels 1 2 3 \
  --ecal-energy-transform log1p \
  --tpad-pe-transform log1p \
  --epochs 5 \
  --batch-size 8 \
  --seed 7 \
  --device mps \
  --cache-model-views \
  --no-progress \
  --num-ecal-plots 2 \
  --num-diagnostic-event-displays 3 \
  --output-root outputs/supervisor_demo_transformer_20k \
  --run-name transformer_3e_20k_seed7
```

Use `--device cuda` on a CUDA machine or `--device cpu` without an
accelerator. Do not change the seed, event count, labels, or source path when
resuming an existing run.

## 4. Resume an Interrupted Run

Re-run the corresponding complete training command and append the appropriate
checkpoint argument:

```bash
--resume outputs/supervisor_demo_transformer_20k/transformer_2e_20k_seed7/checkpoints/latest.pt
```

For 3e, use:

```bash
--resume outputs/supervisor_demo_transformer_20k/transformer_3e_20k_seed7/checkpoints/latest.pt
```

`latest.pt` continues the run. `best.pt` is the validation-selected model for
analysis and presentation.

## 5. Build Best-Checkpoint Inspection Bundles

The trainer writes initial plots from the final epoch. The commands below are
the preferred showcase commands because they load `best.pt` and regenerate
diagnostics without retraining.

```bash
python scripts/inspect_hit_classifier_run.py \
  --run-dir outputs/supervisor_demo_transformer_20k/transformer_2e_20k_seed7 \
  --checkpoint best.pt \
  --split val \
  --num-events 9 \
  --device mps
```

```bash
python scripts/inspect_hit_classifier_run.py \
  --run-dir outputs/supervisor_demo_transformer_20k/transformer_2e_20k_seed7 \
  --checkpoint best.pt \
  --split test \
  --num-events 9 \
  --device mps
```

```bash
python scripts/inspect_hit_classifier_run.py \
  --run-dir outputs/supervisor_demo_transformer_20k/transformer_3e_20k_seed7 \
  --checkpoint best.pt \
  --split val \
  --num-events 9 \
  --device mps
```

```bash
python scripts/inspect_hit_classifier_run.py \
  --run-dir outputs/supervisor_demo_transformer_20k/transformer_3e_20k_seed7 \
  --checkpoint best.pt \
  --split test \
  --num-events 9 \
  --device mps
```

Each command writes under:

```text
<run-dir>/inspection/best/<split>/
```

The bundle contains:

- Per-event hit accuracy scatter and histogram.
- Correlations with loss, confidence, entropy, hit count, energy-weighted
  accuracy, centroid separation, normalized shower separation, and geometric
  ambiguity.
- Assignment-ceiling checks for event-wide label permutations, truth-fraction
  ambiguity, and available TPad context.
- Accuracy and energy-weighted accuracy versus shower separation.
- JSON and CSV records for every inspected event.
- Three worst, three median, and three best static and interactive 3D event
  displays.
- `inspection_manifest.json`, which records the checkpoint, split, summary,
  and every generated file.

## 6. Open the Showcase

Open the result index:

```bash
open outputs/supervisor_demo_transformer_20k/DEMO_INDEX.md
```

Open one 2e worst-case interactive event:

```bash
open outputs/supervisor_demo_transformer_20k/transformer_2e_20k_seed7/inspection/best/val/val_representative_events/val_worst_event_5366_interactive.html
```

Open one 3e worst-case interactive event:

```bash
open outputs/supervisor_demo_transformer_20k/transformer_3e_20k_seed7/inspection/best/val/val_representative_events/val_worst_event_12501_interactive.html
```

Open all selected validation event files in Finder:

```bash
open outputs/supervisor_demo_transformer_20k/transformer_2e_20k_seed7/inspection/best/val/val_representative_events
```

```bash
open outputs/supervisor_demo_transformer_20k/transformer_3e_20k_seed7/inspection/best/val/val_representative_events
```

The HTML uses Plotly from its CDN, so the browser needs an internet connection.
In the interactive display:

- Drag to rotate the detector view.
- Scroll or pinch to zoom.
- Hover over a hit for its coordinates, truth, prediction, and correctness.
- Click a legend item to hide or show a trace; double-click to isolate it.
- Use the Plotly mode bar to reset the camera or save the current view.

## 7. Inspect One Chosen Event

Event indices in the plots are global indices in the saved 20,000-event
dataset. To rebuild only event 5366 without overwriting the full inspection
bundle:

```bash
python scripts/inspect_hit_classifier_run.py \
  --run-dir outputs/supervisor_demo_transformer_20k/transformer_2e_20k_seed7 \
  --checkpoint best.pt \
  --split val \
  --event-index 5366 \
  --num-events 1 \
  --device mps \
  --output-dir outputs/supervisor_demo_transformer_20k/single_events/2e_val_event_5366
```

Repeat `--event-index` to inspect several saved-split events in one command.
The inspector rejects an index that is not part of the requested saved split.

## 8. Suggested Five-Minute Presentation

1. Show `run_overview.md`, then the loss and accuracy histories to establish
   the model and training scope.
2. Show the test confusion matrix as the class-level result.
3. Show the validation event-accuracy overview. The broad distribution is the
   motivation for event-level analysis.
4. Show the correlation grid and shower-separation profile. Smaller normalized
   separation means stronger shower overlap and lower observed accuracy.
5. Compare a worst, median, and best interactive event. Rotate each event and
   use the legend to expose truth classes and prediction errors.
6. Finish with the held-out test summary and state that hyperparameter tuning
   is the next experiment, not part of this demo.

Two useful observations from these runs are that 3e classification is harder
than 2e classification, and mean energy-weighted accuracy is higher than mean
unweighted event accuracy. The latter means that, on average, errors are more
concentrated among lower-energy hits than the most energetic hits.

## 9. Output Map

At each run root:

```text
accuracy_history.png
loss_history.png
test_hit_origin_confusion_matrix.png
history.csv
final_metrics.json
run_overview.md
checkpoints/best.pt
checkpoints/latest.pt
```

At each `inspection/best/<split>/` directory:

```text
<split>_event_accuracy_overview.png
<split>_event_diagnostic_correlations.png
<split>_assignment_ceiling_diagnostics.png
<split>_shower_separation_profiles.png
<split>_event_accuracy.json
<split>_event_accuracy.csv
<split>_representative_events.json
<split>_representative_events/*_prediction_errors.png
<split>_representative_events/*_interactive.html
inspection_manifest.json
```

The run-root test confusion matrix is produced after the final training epoch.
The `inspection/best/` event diagnostics and all headline numbers in this guide
come from the validation-selected checkpoint.
