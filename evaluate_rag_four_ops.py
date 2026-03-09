# 评测 RAG 四算子实验生成结果：仅对 elu, clamp_broadcast, power_broadcast, sigmoid 跑 eval_single_runner，写 result.json。
import os
import json
import subprocess
import tempfile
import argparse
from config import top_p
from utils.utils import get_default_model_from_config
from generate_rag_four_ops import RAG_FOUR_OPS

LANGUAGE = "ascendc"


def eval_rag_four_ops(out_dir, skip_existing=False):
    result = {}
    output_file = os.path.join(out_dir, "result.json")
    if os.path.exists(output_file) and skip_existing:
        print(f"[INFO] Already evaluated, see {output_file} (skip_existing=True)")
        return
    if os.path.exists(output_file):
        print(f"[INFO] Result file will be overwritten: {output_file}")

    for op in RAG_FOUR_OPS:
        txt_path = os.path.join(out_dir, f"{op}.txt")
        if not os.path.isfile(txt_path):
            print(f"[WARN] Missing {txt_path}, skip op {op}")
            result[op] = {"compiled": None, "correctness": None, "performance": None, "compile_info": "Missing generated file"}
            continue
        print(f"[INFO] Evaluating op {op}")
        with open(txt_path, "r") as f:
            response_txt = f.read()
        with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".txt") as tf_input, \
             tempfile.NamedTemporaryFile(mode="r", delete=True, suffix=".json") as tf_output:
            tf_input.write(response_txt)
            tf_input.flush()
            try:
                subprocess.run(
                    ["python3", "eval_single_runner.py", tf_input.name, op, LANGUAGE, tf_output.name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=360,
                )
                tf_output.seek(0)
                result_item = json.load(tf_output)
                result[op] = result_item
                print(f"[INFO] {result_item}")
            except subprocess.CalledProcessError as e:
                if "FileNotFoundError" in (e.stderr or ""):
                    print("[FAIL] FileNotFoundError - check project_root_path in config.py")
                    result[op] = {"compiled": None, "correctness": None, "performance": None, "compile_info": "FileNotFoundError"}
                elif e.returncode == -11:
                    print("[FAIL] Segmentation fault")
                    result[op] = {"compiled": True, "correctness": None, "performance": None, "correctness_info": "Segmentation fault"}
                else:
                    print(f"[FAIL] {e.stderr}")
                    result[op] = {"compiled": True, "correctness": None, "performance": None, "correctness_info": "Unknown fault"}
            except subprocess.TimeoutExpired:
                print("[FAIL] run timeout")
                result[op] = {"compiled": True, "correctness": None, "performance": None, "correctness_info": "Timeout fault"}

    with open(output_file, "w") as f:
        print(f"[INFO] Write result to {output_file}")
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG four-ops generation result.")
    _default_model = get_default_model_from_config() or "deepseek-chat"
    parser.add_argument("--model", type=str, default=_default_model, help="Model name (must match the dir used by generate_rag_four_ops).")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if result.json already exists.")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    for run in range(args.runs):
        out_dir = f"output/ascendc/rag_four_ops_with_official_doc/0.0-{top_p}/{model_name}/run{run}"
        if not os.path.isdir(out_dir):
            print(f"[WARN] Dir not found: {out_dir}, skip run{run}")
            continue
        eval_rag_four_ops(out_dir, skip_existing=args.skip_existing)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
