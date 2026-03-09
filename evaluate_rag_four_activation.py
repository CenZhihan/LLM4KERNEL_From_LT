# 评测 RAG 四算子实验生成结果：对 relu, tanh, softplus, softsign 跑 eval_single_runner，写 result.json。
import os
import json
import subprocess
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import top_p
from utils.utils import get_default_model_from_config

RAG_FOUR_OPS = ["relu", "tanh", "softplus", "softsign"]

LANGUAGE = "ascendc"
EVAL_TIMEOUT = 300  # 5分钟

def eval_single_op(op, out_dir, language):
    result = {"compiled": False, "correctness": None, "performance": None}
    txt_path = os.path.join(out_dir, f"{op}.txt")
    if not os.path.isfile(txt_path):
        print(f"[WARN] Missing {txt_path}, skip op {op}")
        result["compile_info"] = "Missing generated file"
        return op, result
    print(f"[INFO] Evaluating op {op}")
    with open(txt_path, "r") as f:
        response_txt = f.read()
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".txt") as tf_input, \
         tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tf_output:
        tf_input.write(response_txt)
        tf_input.flush()
        out_path = tf_output.name
        try:
            try:
                captured_text = subprocess.run(
                    ["python3", "eval_single_runner.py", tf_input.name, op, language, out_path],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=EVAL_TIMEOUT,
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


def eval_rag_four_ops(out_dir, workers=4, skip_existing=False):
    result = {}
    output_file = os.path.join(out_dir, "result.json")
    if os.path.exists(output_file) and skip_existing:
        print(f"[INFO] Already evaluated, see {output_file} (skip_existing=True)")
        return
    if os.path.exists(output_file):
        print(f"[INFO] Result file will be overwritten: {output_file}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(eval_single_op, op, out_dir, LANGUAGE): op for op in RAG_FOUR_OPS}
        for future in as_completed(futures):
            op, result_item = future.result()
            result[op] = result_item

    # Preserve order
    ordered_result = {op: result[op] for op in RAG_FOUR_OPS if op in result}

    with open(output_file, "w") as f:
        print(f"[INFO] Write result to {output_file}")
        json.dump(ordered_result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG four-activation ops generation result.")
    _default_model = get_default_model_from_config() or "deepseek-chat"
    parser.add_argument("--model", type=str, default=_default_model, help="Model name.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if result.json already exists.")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent evaluations (default 4).")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    for run in range(args.runs):
        out_dir = f"output/ascendc/rag_four_activation/0.0-{top_p}/{model_name}/run{run}"
        if not os.path.isdir(out_dir):
            print(f"[WARN] Dir not found: {out_dir}, skip run{run}")
            continue
        eval_rag_four_ops(out_dir, workers=args.workers, skip_existing=args.skip_existing)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()