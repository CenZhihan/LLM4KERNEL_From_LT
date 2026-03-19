from __future__ import annotations

import argparse
from pathlib import Path

from .paths import get_paths
from .scan_kernels import default_manifest_path, write_manifest
from .write_outputs import materialize_from_manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mkb_adapter", description="KernelBench external kernels adapter.")
    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan passing kernels and write manifest.")
    scan.add_argument(
        "--out",
        type=str,
        default="",
        help="Output manifest path (default: <repo_root>/manifests/passing_165.json).",
    )

    mat = sub.add_parser("materialize", help="Generate {op}.txt files for evaluation_parallel.py.")
    mat.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Manifest path (default: <repo_root>/manifests/passing_165.json).",
    )
    mat.add_argument("--language", type=str, default="ascendc")
    mat.add_argument("--strategy", type=str, default="external165")
    mat.add_argument("--temperature", type=str, default="0.0")
    mat.add_argument("--top_p", type=str, default="1.0")
    mat.add_argument("--model_name", type=str, default="external_lab")
    mat.add_argument("--run_id", type=int, default=0)
    mat.add_argument(
        "--mkb_repo_root",
        type=str,
        default="",
        help="MultiKernelBench repo root (where evaluation_parallel.py lives). "
        "If empty, we infer it as two levels above this ZWN_repo package.",
    )
    mat.add_argument("--no_overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    paths = get_paths()
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "scan":
        out = Path(args.out).expanduser() if args.out else default_manifest_path(paths)
        write_manifest(out, paths=paths)
        print(f"[INFO] Wrote manifest: {out}")
        return 0

    if args.cmd == "materialize":
        manifest_path = (
            Path(args.manifest).expanduser()
            if args.manifest
            else default_manifest_path(paths)
        )
        mkb_root = Path(args.mkb_repo_root).expanduser().resolve() if args.mkb_repo_root else None
        if mkb_root is None:
            mkb_root = paths.repo_root.parent.parent
        written = materialize_from_manifest(
            manifest_path=manifest_path,
            mkb_repo_root=mkb_root,
            language=args.language,
            strategy=args.strategy,
            temperature=args.temperature,
            top_p=args.top_p,
            model_name=args.model_name,
            run_id=args.run_id,
            overwrite=not args.no_overwrite,
        )
        print(f"[INFO] Wrote {len(written)} files.")
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

