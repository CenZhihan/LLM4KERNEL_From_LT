from __future__ import annotations

import re


_CUSTOM_SUFFIX = "_custom"


def op_custom_to_snake_op(op_custom: str) -> str:
    """`argmax_over_a_dimension_custom` -> `argmax_over_a_dimension`."""
    if not op_custom.endswith(_CUSTOM_SUFFIX):
        raise ValueError(f"op_custom must end with '{_CUSTOM_SUFFIX}': {op_custom}")
    return op_custom[: -len(_CUSTOM_SUFFIX)]


def infer_op_custom_from_files(custom_dir_name: str, json_filename: str) -> str:
    """Infer op_custom from a `*custom.json` filename."""
    if not json_filename.endswith(".json"):
        raise ValueError(f"Expected json filename, got: {json_filename}")
    base = json_filename[: -len(".json")]
    if not base.endswith(_CUSTOM_SUFFIX):
        raise ValueError(
            f"Expected '*_custom.json' inside {custom_dir_name}, got: {json_filename}"
        )
    return base


def guess_op_custom_from_pybind(python_bind_src: str) -> str | None:
    """Best-effort parse of the exported function name in `op.cpp`.

    Many kernels use:
      m.def(\"<op_custom>\", &...);
    """

    # Conservative: only accept names that look like snake_case with `_custom`.
    m = re.search(r'm\.def\(\s*"([a-z0-9_]+_custom)"\s*,', python_bind_src)
    return m.group(1) if m else None

