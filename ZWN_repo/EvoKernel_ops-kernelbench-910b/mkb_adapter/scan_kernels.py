from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .name_map import infer_op_custom_from_files, op_custom_to_snake_op
from .paths import RepoPaths, get_paths
from .types import KernelEntry, KernelResult, KernelSourcePaths, ManifestVersion


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_single(glob_root: Path, pattern: str) -> Path:
    hits = list(glob_root.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 match for pattern '{pattern}' under {glob_root}, got {len(hits)}"
        )
    return hits[0]


def scan_ops_dir(paths: RepoPaths) -> list[KernelEntry]:
    """Scan `ops-kernelbench-910b/*` and return entries for compiled+correct kernels."""
    ops_root = paths.ops_root
    if not ops_root.is_dir():
        raise FileNotFoundError(f"ops_root not found: {ops_root}")

    entries: list[KernelEntry] = []
    for custom_dir in sorted([p for p in ops_root.iterdir() if p.is_dir()]):
        result_path = custom_dir / "result.json"
        if not result_path.exists():
            continue
        result_raw = _read_json(result_path)
        compiled = bool(result_raw.get("compiled", False))
        correctness = result_raw.get("correctness", None)
        if not compiled or correctness is not True:
            continue

        # Infer op_custom from the only *_custom.json in the directory.
        project_json = _find_single(custom_dir, "*_custom.json")
        op_custom = infer_op_custom_from_files(custom_dir.name, project_json.name)
        snake_op = op_custom_to_snake_op(op_custom)

        # Locate sources in fixed subpaths.
        host_tiling_h = _find_single(custom_dir / "op_host", "*_custom_tiling.h")
        host_operator_cpp = _find_single(custom_dir / "op_host", "*_custom.cpp")
        kernel_cpp = _find_single(custom_dir / "op_kernel", "*_custom.cpp")
        pybind_cpp = custom_dir / "CppExtension" / "csrc" / "op.cpp"
        if not pybind_cpp.exists():
            raise FileNotFoundError(f"Missing pybind source: {pybind_cpp}")

        entry = KernelEntry(
            custom_project_dir=custom_dir.name,
            op_custom=op_custom,
            snake_op=snake_op,
            sources=KernelSourcePaths(
                project_json=project_json,
                host_tiling_h=host_tiling_h,
                host_operator_cpp=host_operator_cpp,
                kernel_cpp=kernel_cpp,
                pybind_cpp=pybind_cpp,
            ),
            result=KernelResult(
                kernel_project_name=str(result_raw.get("kernel_project_name", custom_dir.name)),
                compiled=compiled,
                correctness=correctness,
                raw=result_raw,
            ),
        )
        entries.append(entry)

    return entries


def manifest_dict(
    *,
    version: ManifestVersion,
    entries: list[KernelEntry],
    paths: RepoPaths,
) -> dict[str, Any]:
    return {
        "version": version,
        "repo_root": str(paths.repo_root),
        "ops_root": str(paths.ops_root),
        "num_entries": len(entries),
        "entries": [
            {
                "custom_project_dir": e.custom_project_dir,
                "op_custom": e.op_custom,
                "snake_op": e.snake_op,
                "sources": {k: str(v) for k, v in asdict(e.sources).items()},
                "result": {
                    "kernel_project_name": e.result.kernel_project_name,
                    "compiled": e.result.compiled,
                    "correctness": e.result.correctness,
                    "raw": e.result.raw,
                },
            }
            for e in entries
        ],
    }


def write_manifest(
    manifest_path: Path,
    *,
    version: ManifestVersion = "mkb_adapter_manifest_v1",
    paths: RepoPaths | None = None,
) -> Path:
    """Write passing-kernels manifest json and return its path."""
    paths = paths or get_paths()
    entries = scan_ops_dir(paths)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest_dict(version=version, entries=entries, paths=paths)
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def default_manifest_path(paths: RepoPaths) -> Path:
    return paths.manifests_dir / "passing_165.json"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan ops-kernelbench-910b and write a manifest of passing kernels."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output manifest path (default: <repo_root>/manifests/passing_165.json).",
    )
    args = parser.parse_args(argv)

    paths = get_paths()
    out = Path(args.out).expanduser() if args.out else default_manifest_path(paths)
    write_manifest(out, paths=paths)
    print(f"[INFO] Wrote manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

