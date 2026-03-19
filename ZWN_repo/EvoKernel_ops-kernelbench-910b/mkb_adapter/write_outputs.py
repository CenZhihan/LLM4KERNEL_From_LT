from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .paths import get_paths
from .render_generated_code import RenderedGeneratedCode, render_generated_code
from .types import KernelSourcePaths


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_manifest_entries(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest.entries must be a list")
    for e in entries:
        if not isinstance(e, dict):
            continue
        yield e


def default_output_dir(
    *,
    mkb_repo_root: Path,
    language: str,
    strategy: str,
    temperature: str,
    top_p: str,
    model_name: str,
    run_id: int,
) -> Path:
    # Must align with evaluation_parallel.py path convention:
    # output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}/{op}.txt
    return (
        mkb_repo_root
        / "output"
        / language
        / strategy
        / f"{temperature}-{top_p}"
        / model_name
        / f"run{run_id}"
    )


def write_one(
    out_dir: Path,
    rendered: RenderedGeneratedCode,
    *,
    overwrite: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rendered.snake_op}.txt"
    if out_path.exists() and not overwrite:
        return out_path
    out_path.write_text(rendered.code, encoding="utf-8")
    return out_path


def materialize_from_manifest(
    *,
    manifest_path: Path,
    mkb_repo_root: Path,
    language: str = "ascendc",
    strategy: str = "external165",
    temperature: str = "0.0",
    top_p: str = "1.0",
    model_name: str = "external_lab",
    run_id: int = 0,
    overwrite: bool = True,
) -> list[Path]:
    manifest = load_manifest(manifest_path)
    out_dir = default_output_dir(
        mkb_repo_root=mkb_repo_root,
        language=language,
        strategy=strategy,
        temperature=temperature,
        top_p=top_p,
        model_name=model_name,
        run_id=run_id,
    )

    written: list[Path] = []
    for e in iter_manifest_entries(manifest):
        sources_dict = e.get("sources")
        if not isinstance(sources_dict, dict):
            raise ValueError("Entry missing sources")

        sources = KernelSourcePaths(
            project_json=Path(sources_dict["project_json"]),
            host_tiling_h=Path(sources_dict["host_tiling_h"]),
            host_operator_cpp=Path(sources_dict["host_operator_cpp"]),
            kernel_cpp=Path(sources_dict["kernel_cpp"]),
            pybind_cpp=Path(sources_dict["pybind_cpp"]),
        )
        rendered = render_generated_code(
            custom_project_dir=str(e.get("custom_project_dir", "")),
            sources=sources,
            op_custom=str(e["op_custom"]),
            snake_op=str(e["snake_op"]),
        )
        written.append(write_one(out_dir, rendered, overwrite=overwrite))
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    paths = get_paths()

    p = argparse.ArgumentParser(
        description="Materialize external kernels as MultiKernelBench eval inputs (.txt)."
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=str(paths.manifests_dir / "passing_165.json"),
        help="Manifest json path produced by scan_kernels.py",
    )
    p.add_argument("--language", type=str, default="ascendc")
    p.add_argument("--strategy", type=str, default="external165")
    p.add_argument("--temperature", type=str, default="0.0")
    p.add_argument("--top_p", type=str, default="1.0")
    p.add_argument("--model_name", type=str, default="external_lab")
    p.add_argument("--run_id", type=int, default=0)
    p.add_argument(
        "--mkb_repo_root",
        type=str,
        default="",
        help="MultiKernelBench repo root (where evaluation_parallel.py lives). "
        "Default: auto-detect by going up from this file to the repo root.",
    )
    p.add_argument("--no_overwrite", action="store_true", help="Do not overwrite existing .txt")
    args = p.parse_args(argv)

    mkb_root = Path(args.mkb_repo_root).expanduser().resolve() if args.mkb_repo_root else None
    if mkb_root is None:
        # Default: assume this package lives at:
        #   <mkb_root>/ZWN_repo/EvoKernel_ops-kernelbench-910b/
        # therefore mkb_root is 2 levels up from repo_root.
        mkb_root = paths.repo_root.parent.parent

    written = materialize_from_manifest(
        manifest_path=Path(args.manifest).expanduser(),
        mkb_repo_root=mkb_root,
        language=args.language,
        strategy=args.strategy,
        temperature=args.temperature,
        top_p=args.top_p,
        model_name=args.model_name,
        run_id=args.run_id,
        overwrite=not args.no_overwrite,
    )
    print(f"[INFO] Wrote {len(written)} files into: {written[0].parent if written else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

