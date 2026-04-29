import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from mldmx.datasets.graph_builder import build_ecal_tpad_context_graph


EVENT_FILE_GLOB = "event_*.pt"


def _load_tensor_event(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _event_file_name(event_idx: int) -> str:
    return f"event_{event_idx:06d}.pt"


def save_tensor_event(event_tensors: dict, output_dir, event_idx: int) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _event_file_name(event_idx)
    torch.save(event_tensors, output_path)
    return output_path


def write_manifest(output_dir, metadata: dict, event_files: list[Path]) -> Path:
    output_dir = Path(output_dir)
    manifest = {
        **metadata,
        "event_files": [path.name for path in event_files],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


class ECalTriggerPadTensorDataset(Dataset):
    """Dataset for preprocessed ECal + TriggerPadTracks event tensor files."""

    def __init__(self, root_dir, event_indices=None):
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.metadata = {}

        if self.manifest_path.exists():
            self.metadata = json.loads(self.manifest_path.read_text())
            event_files = [self.root_dir / name for name in self.metadata["event_files"]]
        else:
            event_files = sorted(self.root_dir.glob(EVENT_FILE_GLOB))

        if event_indices is not None:
            wanted = {int(idx) for idx in event_indices}
            event_files = [
                path for path in event_files if int(path.stem.split("_")[-1]) in wanted
            ]

        if not event_files:
            raise ValueError(f"No preprocessed event files found in {self.root_dir}")

        self.event_files = event_files

    def __len__(self):
        return len(self.event_files)

    def __getitem__(self, index):
        return _load_tensor_event(self.event_files[index])


def tensor_event_to_pyg_data(
    event: dict,
    k_ecal: int = 8,
    k_tpad_to_ecal: int = 16,
    use_precomputed_edges: bool = True,
) -> Data:
    if use_precomputed_edges and "edge_index" in event:
        edge_index = event["edge_index"]
    else:
        edge_index = build_ecal_tpad_context_graph(
            event["ecal_pos"],
            event["tpad"],
            k_ecal=k_ecal,
            k_tpad_to_ecal=k_tpad_to_ecal,
        )

    return Data(
        x=event["x"],
        ecal_pos=event["ecal_pos"],
        tpad=event["tpad"],
        edge_index=edge_index,
        ecal_mask=event["ecal_mask"],
        tpad_mask=event["tpad_mask"],
        y=event["y"],
        physical_y=event["physical_y"],
        event_idx=int(event["event_idx"]),
        num_ecal=int(event["ecal_mask"].sum().item()),
        num_tpad=int(event["tpad_mask"].sum().item()),
    )
