from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .name_map import guess_op_custom_from_pybind
from .types import KernelSourcePaths


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _py_string_literal(s: str) -> str:
    """Return a safe Python string literal for arbitrary source text."""
    # json.dumps gives us a double-quoted string with proper escaping.
    return json.dumps(s, ensure_ascii=False)


def _parse_project_json(project_json_src: str) -> dict[str, Any]:
    """Project json is stored as a file containing a JSON array with one object."""
    data = json.loads(project_json_src)
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError("project_json must be a JSON array with one object")
    return data[0]


def _coerce_default_value(attr_type: str, default_value: str | None) -> Any:
    if default_value is None:
        return None
    dv = default_value
    try:
        # Many files store numeric defaults as strings, e.g. "0.1"
        if attr_type in ("float", "double"):
            return float(dv)
        if attr_type in ("int", "int32", "int64"):
            return int(dv)
        if attr_type == "bool":
            if dv.strip().lower() in ("1", "true", "yes"):
                return True
            if dv.strip().lower() in ("0", "false", "no"):
                return False
            # fallback
            return bool(ast.literal_eval(dv))
        # string or others
        return ast.literal_eval(dv) if dv.strip().startswith(("{", "[", "'", '"')) else dv
    except Exception:
        return dv


@dataclass(frozen=True)
class RenderedGeneratedCode:
    op_custom: str
    snake_op: str
    code: str


def _render_model_src(op_custom: str, project_obj: dict[str, Any]) -> str:
    input_desc = project_obj.get("input_desc") or []
    attrs = project_obj.get("attr") or []

    input_names: list[str] = []
    for i, inp in enumerate(input_desc):
        name = (inp or {}).get("name") or f"input{i}"
        # Python identifiers only
        if not isinstance(name, str) or not name:
            name = f"input{i}"
        input_names.append(name)

    attr_fields: list[tuple[str, Any]] = []
    for a in attrs:
        if not isinstance(a, dict):
            continue
        aname = a.get("name")
        atype = a.get("type") or "str"
        if not isinstance(aname, str) or not aname:
            continue
        default = _coerce_default_value(str(atype), a.get("default_value"))
        attr_fields.append((aname, default))

    init_params = ["*args", "**kwargs"]
    # If attrs exist and have defaults, expose them explicitly to make forward call stable.
    for aname, default in attr_fields:
        py_default = repr(default)
        init_params.append(f"{aname}={py_default}")

    # Forward signature: keep it minimal and positional for tensor inputs.
    forward_params = ["self"] + input_names
    # attrs passed as stored on self
    forward_call_args = ", ".join(input_names + [f"self.{aname}" for aname, _ in attr_fields])

    # Important: keep token `custom_ops_lib` for ascend_compile_pipeline string patching.
    model_src = f"""import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, {", ".join(init_params)}):
        super().__init__()
"""
    for aname, _ in attr_fields:
        model_src += f"        self.{aname} = {aname}\n"

    model_src += f"""
    def forward(self, {", ".join(forward_params[1:])}):
        return custom_ops_lib.{op_custom}({forward_call_args})
"""
    return model_src


def render_generated_code(
    *,
    custom_project_dir: str,
    sources: KernelSourcePaths,
    op_custom: str,
    snake_op: str,
) -> RenderedGeneratedCode:
    """Render a MultiKernelBench-compatible `generated_code` text.

    The resulting text is intended to be saved as `{snake_op}.txt` and later read
    by `evaluation_parallel.py` -> `eval_single_runner.py` -> `ascend_compile_pipeline.ascend_compile`.
    """

    project_json_src = _read_text(sources.project_json)
    host_tiling_src = _read_text(sources.host_tiling_h)
    host_operator_src = _read_text(sources.host_operator_cpp)
    kernel_src = _read_text(sources.kernel_cpp)
    python_bind_src = _read_text(sources.pybind_cpp)

    # Sanity check: ensure we are exporting the expected op name.
    guessed = guess_op_custom_from_pybind(python_bind_src)
    if guessed is not None and guessed != op_custom:
        raise ValueError(
            f"{custom_project_dir}: op_custom mismatch. "
            f"From json inferred {op_custom}, but pybind exports {guessed}"
        )

    project_obj = _parse_project_json(project_json_src)
    model_src = _render_model_src(op_custom, project_obj)

    code = "\n".join(
        [
            f"project_json_src = {_py_string_literal(project_json_src)}",
            f"host_tiling_src = {_py_string_literal(host_tiling_src)}",
            f"host_operator_src = {_py_string_literal(host_operator_src)}",
            f"kernel_src = {_py_string_literal(kernel_src)}",
            f"python_bind_src = {_py_string_literal(python_bind_src)}",
            f"model_src = {_py_string_literal(model_src)}",
            "",
        ]
    )
    return RenderedGeneratedCode(op_custom=op_custom, snake_op=snake_op, code=code)


def render_from_custom_dir(custom_dir: Path) -> RenderedGeneratedCode:
    """Convenience wrapper when you already have `.../*Custom/` directory."""
    from .name_map import infer_op_custom_from_files, op_custom_to_snake_op

    project_json = next(custom_dir.glob("*_custom.json"))
    op_custom = infer_op_custom_from_files(custom_dir.name, project_json.name)
    snake_op = op_custom_to_snake_op(op_custom)

    sources = KernelSourcePaths(
        project_json=project_json,
        host_tiling_h=next((custom_dir / "op_host").glob("*_custom_tiling.h")),
        host_operator_cpp=next((custom_dir / "op_host").glob("*_custom.cpp")),
        kernel_cpp=next((custom_dir / "op_kernel").glob("*_custom.cpp")),
        pybind_cpp=custom_dir / "CppExtension" / "csrc" / "op.cpp",
    )
    return render_generated_code(
        custom_project_dir=custom_dir.name,
        sources=sources,
        op_custom=op_custom,
        snake_op=snake_op,
    )

