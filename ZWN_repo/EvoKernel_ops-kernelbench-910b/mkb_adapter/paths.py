from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    """Paths local to `ZWN_repo/EvoKernel_ops-kernelbench-910b/`."""

    repo_root: Path

    @property
    def ops_root(self) -> Path:
        return self.repo_root / "ops-kernelbench-910b"

    @property
    def manifests_dir(self) -> Path:
        return self.repo_root / "manifests"


def detect_repo_root(start: Path | None = None) -> Path:
    """Return the `.../ZWN_repo/EvoKernel_ops-kernelbench-910b` directory path.

    This file lives at `<repo_root>/mkb_adapter/paths.py`, so we can resolve
    the root without depending on external project state.
    """

    here = Path(__file__).resolve()
    root = here.parent.parent
    if start is not None:
        # Allow override (useful for tests), but keep deterministic default.
        root = Path(start).resolve()
    return root


def get_paths(start: Path | None = None) -> RepoPaths:
    return RepoPaths(repo_root=detect_repo_root(start))

