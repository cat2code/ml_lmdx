'''
Minimal end-to-end pipeline for training a GNN on LDMX data.

This script:
    1. Reads events from ROOT files
    2. Converts each event into a graph (nodes + edges)
    3. Assigns a dummy event-level label based on total ECal energy
    4. Trains a small GNN to classify events
    5. Evaluates performance on a validation split

Pipeline:
    ROOT → tensorization → graph building → PyG Data → GNN → loss → training

Assumptions:
    - read_events(...) returns iterable events
    - tensorize_ecal_event(...) returns (x, pos)
    - build_knn_graph(...) constructs edge_index
    - SimpleGNN implements the model

Purpose:
    - Verify that the full ML pipeline works end-to-end
    - Provide a simple baseline before introducing real labels or tasks

Important:
    - Labels are intentionally simplistic (energy threshold)
    - Goal is to test infrastructure, not physics performance
'''

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mldmx.io.root_reader import read_events
from mldmx.datasets.tensorize import tensorize_ecal_event
from mldmx.datasets.graph_builder import build_knn_graph
from mldmx.models.simple_gnn import SimpleGNN


def make_dummy_event_label(x: torch.Tensor, energy_index: int = 3, threshold: float = 500.0) -> torch.Tensor:
    total_energy = x[:, energy_index].sum()
    return torch.tensor([1 if total_energy > threshold else 0], dtype=torch.long)


def event_to_data(event) -> Data:
    # x: [num_nodes, num_features]
    # pos: [num_nodes, 3]
    x, pos = tensorize_ecal_event(event)

    edge_index = build_knn_graph(pos, k=8)

    y = make_dummy_event_label(x)

    return Data(x=x, pos=pos, edge_index=edge_index, y=y)

def small_io_test():
    from pathlib import Path
    from mldmx.io.root_reader import read_events

    events = read_events(Path("mldmx/data/overlay_main10_pileup20_00/events.root"), max_events=2)

    print(len(events))
    print(events[0].keys())
    print(events[0]["x"].shape)
    print(events[0]["energy"][:5])

def training_test(): # Test 
    root_path = Path("data/overlay_main10_pileup20_00/events.root")

    events = read_events(root_path, max_events=100)
    dataset = [event_to_data(event) for event in events]

    n_train = int(0.8 * len(dataset))
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:]

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleGNN(in_dim=dataset[0].x.shape[1], hidden_dim=32, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                pred = logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_acc={val_acc:.3f}")


def main():
    training_test()


if __name__ == "__main__":
    main()