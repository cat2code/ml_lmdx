# mldmx/src/mldmx/io/root_reader.py

from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import torch
import uproot

from mldmx.io.branches import get_all_branch_names

'''
This module provides utilities for reading ROOT files using uproot with awkward arrays. 
It defines a RootSource dataclass to specify the file and tree, and a read_branches function to read specified branches into awkward arrays.
'''

@dataclass(frozen=True)
class RootSource:
    path: str
    tree_name: str = "LDMX_Events"


def read_branches(source: RootSource, branch_names, entry_start=0, entry_stop=None):
    with uproot.open(source.path) as f:
        tree = f[source.tree_name]
        arrays = tree.arrays(branch_names, entry_start=entry_start, entry_stop=entry_stop, library="ak")
    return arrays

def read_events(root_path, max_events=10):
    """
    Read ECal hit-level information from an LDMX ROOT file and return a list
    of per-event dictionaries.

    Each event has the form:
        {
            "x":      Tensor [N_hits],
            "y":      Tensor [N_hits],
            "z":      Tensor [N_hits],
            "energy": Tensor [N_hits],
        }

    The function first tries the overlay rechit branch layout used in events.root.
    If that fails, it falls back to the simhit layout used in pileup.root.
    """

    root_path = Path(root_path)
    source = RootSource(path=str(root_path), tree_name="LDMX_Events")

    try:
        branch_type = "rechits_overlay"
        branch_names = get_all_branch_names("ecal", branch_type)
        arrays = read_branches(
            source,
            branch_names=branch_names,
            entry_start=0,
            entry_stop=max_events,
        )
    except Exception:
        branch_type = "simhits_pileup"
        branch_names = get_all_branch_names("ecal", branch_type)
        arrays = read_branches(
            source,
            branch_names=branch_names,
            entry_start=0,
            entry_stop=max_events,
        )

    # Exact branch names inferred from your smoke test / branch config
    x_branch = branch_names[0]
    y_branch = branch_names[1]
    z_branch = branch_names[2]
    energy_branch = branch_names[3]

    num_events = len(arrays[x_branch])
    events = []

    for i in range(num_events):
        event = {
            "x": torch.tensor(ak.to_list(arrays[x_branch][i]), dtype=torch.float32),
            "y": torch.tensor(ak.to_list(arrays[y_branch][i]), dtype=torch.float32),
            "z": torch.tensor(ak.to_list(arrays[z_branch][i]), dtype=torch.float32),
            "energy": torch.tensor(ak.to_list(arrays[energy_branch][i]), dtype=torch.float32),
        }
        events.append(event)

    return events