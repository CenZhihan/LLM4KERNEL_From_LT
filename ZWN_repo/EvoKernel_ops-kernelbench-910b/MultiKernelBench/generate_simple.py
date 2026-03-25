#!/usr/bin/env python3
"""
简单的 AscendC 算子生成脚本，使用 requests 直接调用 API，避免 openai 库的依赖问题
"""
import os
import sys
import json
import requests

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 从 api_config 加载配置
try:
    import api_config
    API_KEY = api_config.XI_AI_API_KEY
    BASE_URL = api_config.XI_AI_BASE_URL
except ImportError:
    print("Error: api_config.py not found")
    sys.exit(1)

# 导入数据集和相关模块（这些不依赖 openai）
from dataset import dataset
from utils.utils import read_file, underscore_to_pascalcase

# Prompt 模板
ASCENDC_PROBLEM_STATEMENT = 'You are an expert in writing custom AscendC kernels to optimize PyTorch architectures by replacing specific operators for performance gains.\n'
ASCENDC_PROBLEM_INSTRUCTION = '''
Your task: Replace relevant PyTorch operators in the architecture named Model with custom AscendC kernels. Generate an optimized version named ModelNew, including the six Python strings listed above. Just output the code, no other text, and NO testing code!\n
'''

def ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, example_op):
    op = op + '_custom'
    example_op = example_op + '_custom'
    prompt = ASCENDC_PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
    Here is an example to illustrate the expected transformation using custom AscendC operators. **Original architecture with kernel name `{example_op}`:**\n
    ```python \n
    {example_arch_src}
    ``` \n
    Transformed version using custom AscendC kernels:
    This transformation includes six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src` and `model_src`.
    The kernel function name in `kernel_src` must exactly match the provided kernel name. The operator definition in `project_json_src` and `host_operator_src` should also correspond to the kernel name, but follow PascalCase naming: 
    ```python
    {example_new_arch_src}
    ``` \n
    """

    prompt += f"""
    Now, you are given the following architecture with kernel name {op}(PascalCase: {underscore_to_pascalcase(op)}): \n
    ```python
    {arc_src}
    ```
        """
    prompt += ASCENDC_PROBLEM_INSTRUCTION
    return prompt

def get_prompt(op):
    """生成 prompt"""
    category = dataset[op]['category']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_arch_path = os.path.join(base_dir, "prompts/cuda_model_add.py")
    example_new_arch_path = os.path.join(base_dir, "prompts/ascendc_new_model_add.py")
    new_arch_path = os.path.join(base_dir, f"reference/{category}/{op}.py")
    
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    arch = read_file(new_arch_path)
    
    return ascendc_template(arch, example_arch, example_new_arch, op, 'add')

def call_llm(prompt, model="gpt-5"):
    """使用 requests 直接调用 LLM API"""
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=300)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-5', help='Model name')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='Categories')
    parser.add_argument('--ops', nargs='+', default=None, help='Specific ops to generate')
    args = parser.parse_args()
    
    # 获取要生成的算子
    if args.ops:
        ops_to_generate = args.ops
    else:
        ops_to_generate = [op for op in dataset.keys() 
                          if dataset[op]['category'] in args.categories]
    
    print(f"Will generate {len(ops_to_generate)} operators")
    print(f"Operators: {ops_to_generate}")
    
    # 输出目录
    out_dir = f"output/ascendc/add_shot/0.0-1.0/{args.model}/run0"
    os.makedirs(out_dir, exist_ok=True)
    
    for op in ops_to_generate:
        out_file = os.path.join(out_dir, f"{op}.txt")
        if os.path.exists(out_file):
            print(f"[SKIP] {op} already exists")
            continue
        
        print(f"[INFO] Generating {op}...")
        try:
            prompt = get_prompt(op)
            print(f"[INFO] Prompt length: {len(prompt)} chars")
            response = call_llm(prompt, args.model)
            with open(out_file, 'w') as f:
                f.write(response)
            print(f"[OK] {op} saved to {out_file}")
        except Exception as e:
            print(f"[FAIL] {op}: {e}")

if __name__ == "__main__":
    main()