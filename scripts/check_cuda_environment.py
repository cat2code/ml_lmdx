"""Print a small CUDA/PyTorch environment report."""

import platform
import sys

import torch


def main():
    print(f"python: {sys.version.replace(chr(10), ' ')}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda version: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        print(f"current cuda device: {index}")
        print(f"gpu name: {torch.cuda.get_device_name(index)}")
        x = torch.arange(8, device="cuda", dtype=torch.float32)
        y = (x * x).sum()
        torch.cuda.synchronize()
        print(f"tiny cuda tensor check: {float(y.item()):.1f}")
    else:
        x = torch.arange(8, dtype=torch.float32)
        print(f"tiny cpu tensor check: {float((x * x).sum().item()):.1f}")


if __name__ == "__main__":
    main()
