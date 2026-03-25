#!/usr/bin/env python3
"""
简单的 AscendC 算子评测脚本
"""
import os
import sys
import json
import subprocess
import tempfile

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import dataset

# 导入评测相关模块
from utils.evaluation_utils import eval_single


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-5', help='Model name')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='Categories')
    parser.add_argument('--ops', nargs='+', default=None, help='Specific ops to evaluate')
    parser.add_argument('--strategy', default='add_shot', help='Strategy')
    args = parser.parse_args()
    
    # 获取要评测的算子
    if args.ops:
        ops_to_eval = args.ops
    else:
        ops_to_eval = [op for op in dataset.keys() 
                      if dataset[op]['category'] in args.categories]
    
    print(f"Will evaluate {len(ops_to_eval)} operators")
    
    # 输入目录
    in_dir = f"output/ascendc/{args.strategy}/0.0-1.0/{args.model}/run0"
    
    # 输出文件
    if args.categories == ['all']:
        output_file = os.path.join(in_dir, 'result.json')
    else:
        output_file = os.path.join(in_dir, f'result_{"_".join(args.categories)}.json')
    
    # 加载已有结果（支持增量评测）
    result = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            result = json.load(f)
        print(f"[INFO] Loaded {len(result)} existing results from {output_file}")
    
    language = 'ascendc'
    
    for op in ops_to_eval:
        # 跳过已评测的算子
        if op in result:
            print(f"[SKIP] {op} already evaluated")
            continue
        
        print(f"[INFO] Evaluating {op}...")
        in_file = os.path.join(in_dir, f'{op}.txt')
        
        if not os.path.exists(in_file):
            print(f"[SKIP] {op}.txt not found")
            continue
        
        with open(in_file, 'r') as f:
            response_txt = f.read()
        
        # 使用临时文件传递数据
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tf_input, \
            tempfile.NamedTemporaryFile(mode='r', delete=True) as tf_output:
            
            tf_input.write(response_txt)
            tf_input.flush()
            
            try:
                captured = subprocess.run(
                    ['python3', 'eval_single_runner.py', tf_input.name, op, language, tf_output.name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                result_item = json.load(tf_output)
                
                # 提取编译错误信息
                if not result_item['compiled']:
                    detailed_error = '\n'
                    for line in captured.stdout.split('\n'):
                        if '[ERROR]' in line or 'error:' in line:
                            detailed_error += line + '\n'
                    for line in captured.stderr.split('\n'):
                        if '[ERROR]' in line or 'error:' in line:
                            detailed_error += line + '\n'
                    result_item['compile_info'] += detailed_error
                
            except subprocess.CalledProcessError as e:
                if e.returncode == -11:
                    print(f"[FAIL] Segmentation fault")
                    result[op] = {'compiled': True, 'correctness': None, 'performance': None, 'correctness_info': 'Segmentation fault'}
                else:
                    print(f"[FAIL] Error: {e.stderr}")
                    result[op] = {'compiled': True, 'correctness': None, 'performance': None, 'correctness_info': 'Unknown error'}
                continue
            except subprocess.TimeoutExpired:
                print(f"[FAIL] Timeout")
                result[op] = {'compiled': True, 'correctness': None, 'performance': None, 'correctness_info': 'Timeout'}
                continue
            
            result[op] = result_item
            print(f"[OK] {op}: compiled={result_item['compiled']}, correctness={result_item.get('correctness')}")
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Results saved to {output_file}")


if __name__ == "__main__":
    main()