from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventRef:
    """Reference to a single event inside a file."""

    file_path: str
    tree_name: str
    entry: int
    label: int | None = None

    @property
    def path(self) -> Path:
        return Path(self.file_path)