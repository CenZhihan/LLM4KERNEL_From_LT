#!/usr/bin/env python3
"""
重跑「上次评测因 Timeout 未通过」的算子，使用更长的超时时间，检查是否实际能通过正确性。
实现与 evaluation_parallel.py 一致（同一套 subprocess 并行），仅超时和 op 列表不同。
用法（在项目根目录）:
  python3 eval_timeout_ops.py --result_json output/ascendc/add_shot/0.0-1.0/gpt-5/run0/result.json --timeout 3600
  python3 eval_timeout_ops.py --timeout 3600 --output rerun_timeout_result.json --workers 8
"""
import os
import json
import subprocess
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_timeout_ops(result_json_path):
    """从 result.json 中找出 compiled=True 且 correctness_info 为 Timeout 的 op 列表。"""
    with open(result_json_path) as f:
        result = json.load(f)
    timeout_ops = [
        op for op, v in result.items()
        if v.get("compiled") is True
        and (v.get("correctness_info") or "").strip().lower().startswith("timeout")
    ]
    return timeout_ops


def eval_single_op(op, out_dir, language, timeout_sec):
    """
    与 evaluation_parallel.eval_single_op 一致，仅 timeout 可配。
    Run evaluation for one op. Returns (op, result_dict).
    """
    result = {"compiled": False, "correctness": None, "performance": None}
    op_path = os.path.join(out_dir, f"{op}.txt")
    if not os.path.exists(op_path):
        print(f"[FAIL] op {op}: missing {op_path}")
        result["correctness_info"] = "Missing generated file"
        return op, result
    try:
        with open(op_path, "r") as saved_log:
            response_txt = saved_log.read()
    except Exception as e:
        result["correctness_info"] = str(e)
        return op, result

    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".txt") as tf_input, \
            tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tf_out:
        tf_input.write(response_txt)
        tf_input.flush()
        out_path = tf_out.name
        try:
            try:
                captured_text = subprocess.run(
                    ["python3", "eval_single_runner.py", tf_input.name, op, language, out_path],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                with open(out_path, "r") as f:
                    result_item = json.load(f)
                if not result_item.get("compiled", True):
                    detailed_compiler_error = "\n"
                    for line in (captured_text.stdout or "").split("\n"):
                        if "[ERROR]" in line or "error:" in line:
                            detailed_compiler_error += line + "\n"
                    for line in (captured_text.stderr or "").split("\n"):
                        if "[ERROR]" in line or "error:" in line:
                            detailed_compiler_error += line + "\n"
                    result_item["compile_info"] = result_item.get("compile_info", "") + detailed_compiler_error
                print(f"[INFO] Evaluating op {op} -> {result_item}")
                return op, result_item
            except subprocess.CalledProcessError as e:
                if "FileNotFoundError" in (e.stderr or ""):
                    print(f"[FAIL] op {op}: FileNotFoundError - check project_root_path in config.py")
                    result["correctness_info"] = "FileNotFoundError"
                elif e.returncode == -11:
                    print(f"[FAIL] op {op}: Segmentation fault")
                    result = {"compiled": True, "correctness": None, "performance": None, "correctness_info": "Segmentation fault"}
                else:
                    print(f"[FAIL] op {op}: {e.stderr}")
                    result = {"compiled": True, "correctness": None, "performance": None, "correctness_info": "Unknown fault"}
                return op, result
            except subprocess.TimeoutExpired:
                print(f"[FAIL] op {op}: run timeout")
                result = {"compiled": True, "correctness": None, "performance": None, "correctness_info": "Timeout fault"}
                return op, result
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Re-run evaluation for ops that previously failed due to timeout, with a longer timeout."
    )
    parser.add_argument(
        "--result_json",
        type=str,
        default="output/ascendc/add_shot/0.0-1.0/gpt-5/run0/result.json",
        help="Path to result.json from which to extract timeout ops.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each op evaluation (default: 3600).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ascendc",
        help="Backend language (default: ascendc).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save re-run results JSON. If not set, only prints summary.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory containing {op}.txt. Default: same dir as result_json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max concurrent evaluations (default: 4).",
    )
    args = parser.parse_args()

    result_json = os.path.abspath(args.result_json)
    out_dir = os.path.abspath(args.out_dir or os.path.dirname(result_json))

    timeout_ops = get_timeout_ops(result_json)
    if not timeout_ops:
        print("[INFO] No timeout ops found in", args.result_json)
        return

    print(f"[INFO] Found {len(timeout_ops)} timeout ops. Re-running with timeout={args.timeout}s, workers={args.workers}.")
    print(f"[INFO] out_dir={out_dir}")
    print()

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(eval_single_op, op, out_dir, args.language, args.timeout): op
            for op in timeout_ops
        }
        for future in as_completed(futures):
            op = futures[future]
            op, result_item = future.result()
            results[op] = result_item

    # 按原 timeout_ops 顺序
    ordered_result = {op: results[op] for op in timeout_ops if op in results}

    n_pass = sum(1 for v in ordered_result.values() if v.get("correctness") is True)
    n_timeout = sum(1 for v in ordered_result.values() if "timeout" in (v.get("correctness_info") or "").lower())
    n_other = len(ordered_result) - n_pass - n_timeout
    print("=== Summary ===")
    print(f"  Total re-run: {len(ordered_result)}")
    print(f"  Pass (correctness=True): {n_pass}")
    print(f"  Still timeout: {n_timeout}")
    print(f"  Other fail: {n_other}")

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w") as f:
            json.dump(ordered_result, f, indent=2)
        print(f"\n[INFO] Results written to {out_path}")


if __name__ == "__main__":
    main()
