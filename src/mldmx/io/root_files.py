import math
import re
from pathlib import Path


def root_file_sort_key(path):
    path = Path(path)
    match = re.search(r"events_(\d+)\.root$", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, math.inf, path.name)


def find_root_files(data_dir):
    data_dir = Path(data_dir)
    root_files = sorted(data_dir.glob("events_*.root"), key=root_file_sort_key)
    if not root_files:
        root_files = sorted(data_dir.glob("*.root"), key=root_file_sort_key)
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in {data_dir}")
    return root_files
