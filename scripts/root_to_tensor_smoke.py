import argparse
from pathlib import Path
import torch
import torch.nn as nn

from mldmx.io.root_reader import RootSource, read_branches
from mldmx.io.branches import get_vector_branches, get_all_branch_names
from mldmx.datasets.tensorize import ecal_hits_to_padded_tensor

"""
Smoke test:
python3 scripts/root_to_tensor_smoke.py data/overlay_main10_pileup20_00/pileup.root --stop 5
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=10)
    parser.add_argument("--max-hits", type=int, default=256)
    args = parser.parse_args()

    root_path = Path(args.root_file).resolve()
    print(f"Opening ROOT file: {root_path}")

    if not root_path.exists():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    source = RootSource(path=str(root_path), tree_name="LDMX_Events")
    
    try: # Read rechits branches (events.root file)
        branch_type = "rechits_overlay"
        vectors = get_vector_branches("ecal", branch_type)
        branch_names = get_all_branch_names("ecal", branch_type)

    except: # Read simhits branches instead (pileup.root file)
        print("Reading branch failed, looking for simhits branch instead...")
        branch_type = "simhits_pileup"
        vectors = get_vector_branches("ecal", branch_type)
        branch_names = get_all_branch_names("ecal", branch_type)

    print("Reading branches:")
    for b in branch_names:
        print("  ", b)

    arrays = read_branches(
        source,
        branch_names=branch_names,
        entry_start=args.start,
        entry_stop=args.stop,
    )
        
        

    X, mask = ecal_hits_to_padded_tensor(arrays, vectors, max_hits=args.max_hits)

    print("X shape:", X.shape)
    print("mask shape:", mask.shape)
    print("valid hits per event:", mask.sum(dim=1).tolist())

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )

    out = model(X)
    print("model output shape:", out.shape)


if __name__ == "__main__":
    main()