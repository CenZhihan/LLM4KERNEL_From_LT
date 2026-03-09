# RAG 实验：仅对 4 个基础 Activation 算子，用华为官方 API 文档拼入 prompt 后生成。
# 4 个算子: relu, tanh, softplus, softsign
import os
import argparse
from utils.utils import get_client, get_default_model_from_config
from config import temperature, num_completions, top_p, ascendc_official_api_doc_path
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template
from generate_and_write import generate_and_write_single

RAG_FOUR_OPS = ["relu", "tanh", "softplus", "softsign"]

# 官方文档最大字符数，避免 context 爆掉
OFFICIAL_API_DOC_MAX_CHARS = 12000


def _load_official_api_doc():
    if not ascendc_official_api_doc_path or not os.path.isfile(ascendc_official_api_doc_path):
        return ""
    try:
        with open(ascendc_official_api_doc_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return ""
    content = content.strip()
    if not content:
        return ""
    if len(content) > OFFICIAL_API_DOC_MAX_CHARS:
        content = content[:OFFICIAL_API_DOC_MAX_CHARS] + "\n\n(... 已截断，超过长度限制 ...)"
    return content


def generate_prompt_rag_four_activation(op):
    """与 add_shot 相同的 base prompt，再在开头拼入华为官方 API 文档。"""
    arc_src, example_arch_src, example_new_arch_src = read_relavant_files("ascendc", op, "add")
    base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, "add")
    doc_content = _load_official_api_doc()
    if not doc_content:
        return base_prompt
    doc_section = "## 华为官方 Ascend C API 参考\n\n" + doc_content + "\n\n---\n\n"
    return doc_section + base_prompt


def main():
    parser = argparse.ArgumentParser(description="Generate kernel for RAG four-activation ops with Huawei official API doc.")
    _default_model = get_default_model_from_config() or "deepseek-chat"
    parser.add_argument("--model", type=str, default=_default_model, help="Model name.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    args = parser.parse_args()

    model = args.model
    runs = args.runs
    model_name = model.split("/")[-1] if "/" in model else model

    doc_loaded = _load_official_api_doc()
    print(f"[INFO] Official API doc loaded: {len(doc_loaded)} chars" if doc_loaded else "[INFO] No official API doc file or empty, prompt will not include doc.")

    for run in range(runs):
        out_dir = f"output/ascendc/rag_four_activation/0.0-{top_p}/{model_name}/run{run}"
        os.makedirs(out_dir, exist_ok=True)
        for op in RAG_FOUR_OPS:
            if os.path.exists(os.path.join(out_dir, f"{op}.txt")):
                print(f"[INFO] Already generated at {out_dir}/{op}.txt, skip")
                continue
            print(f"[INFO] Generate kernel for op {op} (RAG four-activation with official doc)")
            prompt = generate_prompt_rag_four_activation(op)
            client = get_client(model)
            try:
                generate_and_write_single(prompt, client, out_dir, op, model)
            except Exception as e:
                print(f"[FAIL] op {op}: {e}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()