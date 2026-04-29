"""
Preprocess labelled ECal RecHits plus TriggerPadTracks context into per-event tensors.

Run from the mldmx directory:

    python3 scripts/preprocess_ecal_tpad_dataset.py
"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from mldmx.datasets.ecal_tpad_dataset import save_tensor_event, write_manifest
from mldmx.datasets.graph_builder import build_ecal_tpad_context_graph
from mldmx.datasets.tensorize import tensorize_ecal_with_triggerpad_context
from mldmx.io.root_reader import read_ecal_rechits_with_truth_and_triggerpad_context


VALID_LABELS = (1, 2, 3)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-file",
        default=project_root / "data/28apr_00/events.root",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=project_root / "data/processed/ecal_tpad_3class",
        type=Path,
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--k-ecal", type=int, default=8)
    parser.add_argument("--k-tpad-to-ecal", type=int, default=16)
    parser.add_argument("--keep-noise", action="store_true")
    parser.add_argument(
        "--no-edge-index",
        action="store_true",
        help="Store node tensors only; graph edges can be rebuilt at training time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_file = args.root_file.resolve()
    output_dir = args.output_dir.resolve()
    filter_noise = not args.keep_noise

    print(f"Reading ROOT file: {root_file}")
    events = read_ecal_rechits_with_truth_and_triggerpad_context(
        root_file,
        max_events=args.max_events,
    )
    print(f"number of events: {len(events)}")
    print(f"output directory: {output_dir}")
    print(
        "Noise handling: "
        + ("filtering out noise hits" if filter_noise else "keeping noise hits")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    event_files = []
    class_counts = Counter()
    total_ecal = 0
    total_tpad = 0

    for event_idx, event in enumerate(events):
        tensors = tensorize_ecal_with_triggerpad_context(
            event,
            valid_labels=VALID_LABELS,
            filter_noise=filter_noise,
        )
        tensor_event = {
            "x": tensors["x"],
            "ecal_pos": tensors["ecal_pos"],
            "tpad": tensors["tpad"],
            "ecal_mask": tensors["ecal_mask"],
            "tpad_mask": tensors["tpad_mask"],
            "y": tensors["y"],
            "physical_y": tensors["physical_y"],
            "event_idx": torch.tensor(event_idx, dtype=torch.long),
        }
        if not args.no_edge_index:
            tensor_event["edge_index"] = build_ecal_tpad_context_graph(
                tensors["ecal_pos"],
                tensors["tpad"],
                k_ecal=args.k_ecal,
                k_tpad_to_ecal=args.k_tpad_to_ecal,
            )

        output_path = save_tensor_event(tensor_event, output_dir, event_idx)
        event_files.append(output_path)

        num_ecal = int(tensors["ecal_mask"].sum().item())
        num_tpad = int(tensors["tpad_mask"].sum().item())
        total_ecal += num_ecal
        total_tpad += num_tpad
        class_counts.update(tensors["physical_y"].tolist())
        print(
            f"event {event_idx}: selected_ecal_hits={num_ecal}, "
            f"tpad_nodes={num_tpad}, saved={output_path.name}"
        )

    unique_labels = sorted(class_counts)
    if unique_labels != list(VALID_LABELS):
        raise ValueError(
            f"Expected physical labels {VALID_LABELS}, but saw {unique_labels}."
        )

    manifest_path = write_manifest(
        output_dir,
        metadata={
            "root_file": str(root_file),
            "num_events": len(event_files),
            "valid_labels": list(VALID_LABELS),
            "filter_noise": filter_noise,
            "feature_layout": [
                "is_ecal",
                "is_tpad",
                "ecal_x",
                "ecal_y",
                "ecal_z",
                "ecal_energy",
                "tpad_centroid",
                "tpad_pe",
            ],
            "edge_index_stored": not args.no_edge_index,
            "k_ecal": args.k_ecal if not args.no_edge_index else None,
            "k_tpad_to_ecal": args.k_tpad_to_ecal if not args.no_edge_index else None,
            "class_counts": dict(sorted(class_counts.items())),
            "total_selected_ecal_hits": total_ecal,
            "total_tpad_nodes": total_tpad,
        },
        event_files=event_files,
    )
    print(f"wrote manifest: {manifest_path}")
    print("class counts:", dict(sorted(class_counts.items())))
    print(f"total selected ECal hits: {total_ecal}")
    print(f"total TriggerPadTracks nodes: {total_tpad}")


if __name__ == "__main__":
    main()
