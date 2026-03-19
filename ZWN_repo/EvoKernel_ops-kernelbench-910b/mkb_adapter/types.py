from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class KernelResult:
    kernel_project_name: str
    compiled: bool
    correctness: bool | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class KernelSourcePaths:
    project_json: Path
    host_tiling_h: Path
    host_operator_cpp: Path
    kernel_cpp: Path
    pybind_cpp: Path


@dataclass(frozen=True)
class KernelEntry:
    custom_project_dir: str
    op_custom: str
    snake_op: str
    sources: KernelSourcePaths
    result: KernelResult


ManifestVersion = Literal["mkb_adapter_manifest_v1"]

