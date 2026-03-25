#!/usr/bin/env python3
"""
简单的 AscendC 算子生成脚本，不依赖复杂的包
"""
import os
import sys
import json

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接使用 requests 调用 API
import requests

# 从 api_config 加载配置
try:
    import api_config
    API_KEY = api_config.XI_AI_API_KEY
    BASE_URL = api_config.XI_AI_BASE_URL
except ImportError:
    print("Error: api_config.py not found")
    sys.exit(1)

# 导入数据集
from dataset import dataset

# Prompt 生成相关
from prompt_generators.prompt_utils import ascendc_template
from utils.utils import read_file

def get_prompt(op):
    """生成 prompt"""
    category = dataset[op]['category']
    example_arch_path = os.path.join(os.path.dirname(__file__), "prompts/cuda_model_add.py")
    example_new_arch_path = os.path.join(os.path.dirname(__file__), "prompts/ascendc_new_model_add.py")
    new_arch_path = os.path.join(os.path.dirname(__file__), f"reference/{category}/{op}.py")
    
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    arch = read_file(new_arch_path)
    
    return ascendc_template(arch, example_arch, example_new_arch, op, 'add')

def call_llm(prompt, model="gpt-5"):
    """调用 LLM API"""
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
    
    print(f"Will generate {len(ops_to_generate)} operators: {ops_to_generate}")
    
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
            response = call_llm(prompt, args.model)
            with open(out_file, 'w') as f:
                f.write(response)
            print(f"[OK] {op} saved to {out_file}")
        except Exception as e:
            print(f"[FAIL] {op}: {e}")

if __name__ == "__main__":
    main()