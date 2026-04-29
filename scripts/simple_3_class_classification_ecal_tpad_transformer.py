"""
Run from the mldmx directory:

    cd mldmx
    python3 -m pip install -e .
    python3 scripts/simple_3_class_classification_ecal_tpad_transformer.py
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from mldmx.datasets.ecal_tpad_dataset import ECalTriggerPadTensorDataset
from mldmx.datasets.tensorize import tensorize_ecal_with_triggerpad_context
from mldmx.io.root_reader import read_ecal_rechits_with_truth_and_triggerpad_context
from mldmx.models import ECalHitTransformer
from mldmx.viz.ecal import plot_ecal_hit_classes_3d


VALID_LABELS = (1, 2, 3)


def event_to_tensors(event, event_idx, filter_noise=True):
    tensors = tensorize_ecal_with_triggerpad_context(
        event,
        valid_labels=VALID_LABELS,
        filter_noise=filter_noise,
    )
    tensors["event_idx"] = event_idx
    return tensors


def count_classes(events):
    counter = Counter()
    for event in events:
        counter.update(event["physical_y"].tolist())
    return dict(sorted(counter.items()))


def choose_device(requested_device):
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-file",
        default=project_root / "data/28apr_00/events.root",
        type=Path,
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        type=Path,
        help="Optional directory produced by preprocess_ecal_tpad_dataset.py.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help="Use CPU by default for reproducible smoke tests; pass auto to use CUDA/MPS if available.",
    )
    parser.add_argument("--keep-noise", action="store_true")
    return parser.parse_args()


def load_tensor_events(args):
    if args.processed_dir is not None:
        processed_dir = args.processed_dir.resolve()
        print(f"Reading preprocessed dataset: {processed_dir}")
        dataset = ECalTriggerPadTensorDataset(processed_dir)
        tensor_events = [dataset[i] for i in range(len(dataset))]
        print(f"number of events: {len(tensor_events)}")
        for event in tensor_events:
            event_idx = int(event["event_idx"])
            print(
                f"event {event_idx}: selected_ecal_hits={event['ecal_mask'].sum().item()}, "
                f"tpad_nodes={event['tpad_mask'].sum().item()}, "
                f"labels={sorted(set(event['physical_y'].tolist()))}"
            )
        return tensor_events

    root_file = args.root_file.resolve()
    print(f"Reading ROOT file: {root_file}")
    events = read_ecal_rechits_with_truth_and_triggerpad_context(root_file, max_events=10)
    print(f"number of events: {len(events)}")

    filter_noise = not args.keep_noise
    tensor_events = []
    for event_idx, event in enumerate(events):
        n_noise = sum(bool(v) for v in event["noise_flag"])
        n_tpad = len(event["trigger_pad_tracks"]["centroid"])
        print(
            f"event {event_idx}: raw ECal hits={len(event['x'])}, "
            f"noise_hits={n_noise}, TriggerPadTracks={n_tpad}"
        )
        tensor_event = event_to_tensors(event, event_idx=event_idx, filter_noise=filter_noise)
        tensor_events.append(tensor_event)
        print(
            f"event {event_idx}: selected_ecal_hits={tensor_event['ecal_mask'].sum().item()}, "
            f"tpad_nodes={tensor_event['tpad_mask'].sum().item()}, "
            f"labels={sorted(set(tensor_event['physical_y'].tolist()))}"
        )

    return tensor_events


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models/simple_3_class_ecal_tpad_transformer.pt"
    pred_plot_path = project_root / "figures/simple_3_class_event9_tpad_transformer_predicted.png"
    truth_plot_path = project_root / "figures/simple_3_class_event9_tpad_transformer_truth.png"

    filter_noise = not args.keep_noise
    print(
        "Noise handling: "
        + ("filtering out noise hits before training/evaluation" if filter_noise else "keeping noise hits")
    )

    tensor_events = load_tensor_events(args)
    if len(tensor_events) < 10:
        raise ValueError(f"Expected at least 10 events, found {len(tensor_events)}.")

    train_events = tensor_events[:9]
    test_event = tensor_events[9]

    unique_labels = sorted({label for event in tensor_events for label in event["physical_y"].tolist()})
    if unique_labels != list(VALID_LABELS):
        raise ValueError(
            f"Expected physical labels {VALID_LABELS}, but saw {unique_labels}. "
            "Check that origin_id_contribs contains the intended 3-class labels."
        )

    print(f"training events: 0-8 ({len(train_events)} events)")
    print("evaluation event: 9")
    print("training class counts:", count_classes(train_events))
    print("unique labels seen:", unique_labels)

    device = choose_device(args.device)
    print(f"device: {device}")
    model = ECalHitTransformer(
        in_dim=train_events[0]["x"].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        out_dim=3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_ecal_hits = 0
        for event in train_events:
            x = event["x"].to(device)
            ecal_mask = event["ecal_mask"].to(device)
            y = event["y"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits[ecal_mask], y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_ecal_hits += y.numel()

        print(f"epoch={epoch:03d} train_loss={total_loss / total_ecal_hits:.4f}")

    model.eval()
    with torch.no_grad():
        x = test_event["x"].to(device)
        ecal_mask = test_event["ecal_mask"].to(device)
        logits = model(x)
        pred_class = logits[ecal_mask].argmax(dim=1).cpu()
        true_class = test_event["y"].cpu()
        accuracy = (pred_class == true_class).float().mean().item()

    class_to_label = {0: 1, 1: 2, 2: 3}
    pred_physical = torch.tensor([class_to_label[int(v)] for v in pred_class], dtype=torch.long)
    true_physical = torch.tensor([class_to_label[int(v)] for v in true_class], dtype=torch.long)
    print(f"event 9 test accuracy: {accuracy:.3f}")
    print("event 9 true class counts:", dict(sorted(Counter(true_physical.tolist()).items())))
    print("event 9 predicted class counts:", dict(sorted(Counter(pred_physical.tolist()).items())))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "valid_labels": VALID_LABELS,
            "class_to_label": class_to_label,
            "model_kwargs": {
                "in_dim": train_events[0]["x"].shape[1],
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "out_dim": 3,
            },
        },
        model_path,
    )
    print(f"saved model: {model_path}")

    pos = test_event["ecal_pos"].cpu()
    plot_ecal_hit_classes_3d(
        pos,
        pred_physical,
        pred_plot_path,
        "Event 9 ECal hits, TriggerPad transformer predicted dominant origin_id",
    )
    plot_ecal_hit_classes_3d(
        pos,
        true_physical,
        truth_plot_path,
        "Event 9 ECal hits, true dominant origin_id",
    )
    print(f"saved prediction plot: {pred_plot_path}")
    print(f"saved truth plot: {truth_plot_path}")


if __name__ == "__main__":
    main()
