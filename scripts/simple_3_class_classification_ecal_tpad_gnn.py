"""
Run from the mldmx directory:

    cd mldmx
    python3 -m pip install -e .
    python3 scripts/simple_3_class_classification_ecal_tpad_gnn.py
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mldmx.datasets.ecal_tpad_dataset import (
    ECalTriggerPadTensorDataset,
    tensor_event_to_pyg_data,
)
from mldmx.datasets.graph_builder import build_ecal_tpad_context_graph
from mldmx.datasets.tensorize import tensorize_ecal_with_triggerpad_context
from mldmx.io.root_reader import read_ecal_rechits_with_truth_and_triggerpad_context
from mldmx.models import ECalTriggerPadGNN
from mldmx.viz.ecal import plot_ecal_hit_classes_3d


VALID_LABELS = (1, 2, 3)


def event_to_graph(event, event_idx, filter_noise=True, k_ecal=8, k_tpad_to_ecal=16):
    tensors = tensorize_ecal_with_triggerpad_context(
        event,
        valid_labels=VALID_LABELS,
        filter_noise=filter_noise,
    )
    edge_index = build_ecal_tpad_context_graph(
        tensors["ecal_pos"],
        tensors["tpad"],
        k_ecal=k_ecal,
        k_tpad_to_ecal=k_tpad_to_ecal,
    )
    return Data(
        x=tensors["x"],
        ecal_pos=tensors["ecal_pos"],
        tpad=tensors["tpad"],
        edge_index=edge_index,
        ecal_mask=tensors["ecal_mask"],
        tpad_mask=tensors["tpad_mask"],
        y=tensors["y"],
        physical_y=tensors["physical_y"],
        event_idx=event_idx,
        num_ecal=int(tensors["ecal_mask"].sum().item()),
        num_tpad=int(tensors["tpad_mask"].sum().item()),
    )


def count_classes(graphs):
    counter = Counter()
    for graph in graphs:
        counter.update(graph.physical_y.tolist())
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
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--k-ecal", type=int, default=8)
    parser.add_argument("--k-tpad-to-ecal", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help="Use CPU by default for reproducible smoke tests; pass auto to use CUDA/MPS if available.",
    )
    parser.add_argument("--keep-noise", action="store_true")
    return parser.parse_args()


def load_graphs(args):
    if args.processed_dir is not None:
        processed_dir = args.processed_dir.resolve()
        print(f"Reading preprocessed dataset: {processed_dir}")
        dataset = ECalTriggerPadTensorDataset(processed_dir)
        graphs = [
            tensor_event_to_pyg_data(
                dataset[i],
                k_ecal=args.k_ecal,
                k_tpad_to_ecal=args.k_tpad_to_ecal,
                use_precomputed_edges=True,
            )
            for i in range(len(dataset))
        ]
        print(f"number of events: {len(graphs)}")
        for graph in graphs:
            print(
                f"event {int(graph.event_idx)}: selected_ecal_hits={graph.num_ecal}, "
                f"tpad_nodes={graph.num_tpad}, labels={sorted(set(graph.physical_y.tolist()))}"
            )
        return graphs

    root_file = args.root_file.resolve()
    print(f"Reading ROOT file: {root_file}")
    events = read_ecal_rechits_with_truth_and_triggerpad_context(root_file, max_events=10)
    print(f"number of events: {len(events)}")

    filter_noise = not args.keep_noise
    graphs = []
    for event_idx, event in enumerate(events):
        n_noise = sum(bool(v) for v in event["noise_flag"])
        n_tpad = len(event["trigger_pad_tracks"]["centroid"])
        print(
            f"event {event_idx}: raw ECal hits={len(event['x'])}, "
            f"noise_hits={n_noise}, TriggerPadTracks={n_tpad}"
        )
        graph = event_to_graph(
            event,
            event_idx=event_idx,
            filter_noise=filter_noise,
            k_ecal=args.k_ecal,
            k_tpad_to_ecal=args.k_tpad_to_ecal,
        )
        graphs.append(graph)
        print(
            f"event {event_idx}: selected_ecal_hits={graph.num_ecal}, "
            f"tpad_nodes={graph.num_tpad}, labels={sorted(set(graph.physical_y.tolist()))}"
        )

    return graphs


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models/simple_3_class_ecal_tpad_gnn.pt"
    pred_plot_path = project_root / "figures/simple_3_class_event9_tpad_gnn_predicted.png"
    truth_plot_path = project_root / "figures/simple_3_class_event9_tpad_gnn_truth.png"

    filter_noise = not args.keep_noise
    print(
        "Noise handling: "
        + ("filtering out noise hits before training/evaluation" if filter_noise else "keeping noise hits")
    )

    graphs = load_graphs(args)
    if len(graphs) < 10:
        raise ValueError(f"Expected at least 10 events, found {len(graphs)}.")

    train_graphs = graphs[:9]
    test_graph = graphs[9]

    unique_labels = sorted({label for graph in graphs for label in graph.physical_y.tolist()})
    if unique_labels != list(VALID_LABELS):
        raise ValueError(
            f"Expected physical labels {VALID_LABELS}, but saw {unique_labels}. "
            "Check that origin_id_contribs contains the intended 3-class labels."
        )

    print(f"training events: 0-8 ({len(train_graphs)} events)")
    print("evaluation event: 9")
    print("training class counts:", count_classes(train_graphs))
    print("unique labels seen:", unique_labels)

    loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    device = choose_device(args.device)
    print(f"device: {device}")
    model = ECalTriggerPadGNN(
        in_dim=train_graphs[0].x.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_ecal_nodes = 0
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits[batch.ecal_mask], batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.y.numel()
            total_ecal_nodes += batch.y.numel()

        print(f"epoch={epoch:03d} train_loss={total_loss / total_ecal_nodes:.4f}")

    model.eval()
    with torch.no_grad():
        test_graph = test_graph.to(device)
        logits = model(test_graph.x, test_graph.edge_index)
        ecal_logits = logits[test_graph.ecal_mask]
        pred_class = ecal_logits.argmax(dim=1).cpu()
        true_class = test_graph.y.cpu()
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
                "in_dim": train_graphs[0].x.shape[1],
                "hidden_dim": args.hidden_dim,
                "out_dim": 3,
            },
        },
        model_path,
    )
    print(f"saved model: {model_path}")

    pos = test_graph.ecal_pos.cpu()
    plot_ecal_hit_classes_3d(
        pos,
        pred_physical,
        pred_plot_path,
        "Event 9 ECal hits, TriggerPad GNN predicted dominant origin_id",
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
