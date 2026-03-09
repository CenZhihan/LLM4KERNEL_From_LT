#!/usr/bin/env python3
"""
并行验证 5 个 AscendC Few-shot 参考样例的环境与硬编码 Tiling 逻辑。
"""
import os
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ENV_TEST_OPS = [
    ("ascendc_new_model_matmul_add.py", "matmul_add"),
    ("ascendc_new_model_leaky_relu.py", "leaky_relu"),
    ("ascendc_new_model_mse_loss.py", "mse_loss"),
    ("ascendc_new_model_layer_norm.py", "layer_norm"),
    ("ascendc_new_model_reduce_sum.py", "reduce_sum")
]

EVAL_TIMEOUT = 360
output_file = 'result_examples.json'

def test_single_op(prompt_file, op_name):
    print(f"[INFO] 正在编译并验证 {op_name} (采用理想的硬编码形状)...")
    
    result_item = {'compiled': False, 'correctness': None, 'correctness_info': None}
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tf_out:
        out_path = tf_out.name
        
    try:
        # 使用 subprocess 调用独立的 runner 避免多个 npu 编译时的全局变量和环境污染
        captured_text = subprocess.run(
            ['python3', 'test_single_shape_runner.py', prompt_file, op_name, out_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=EVAL_TIMEOUT
        )
        
        with open(out_path, 'r') as f:
            result_item = json.load(f)
            
        if not result_item.get('compiled', True):
            detailed_compiler_error = '\n'
            for line in (captured_text.stdout or '').split('\n'):
                if '[ERROR]' in line or 'error:' in line:
                    detailed_compiler_error += line + '\n'
            for line in (captured_text.stderr or '').split('\n'):
                if '[ERROR]' in line or 'error:' in line:
                    detailed_compiler_error += line + '\n'
            result_item['compile_info'] = result_item.get('compile_info', '') + detailed_compiler_error
            
        if result_item.get('correctness'):
            print(f"  [SUCCESS] {op_name} 验证通过: {result_item.get('correctness_info')}")
        else:
            print(f"  [FAIL] {op_name} 验证失败: {result_item.get('correctness_info')}")
            
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {op_name} 运行出错，Exit Code {e.returncode}")
        detailed_error = '\n'
        for line in (e.stdout or '').split('\n'):
            if '[ERROR]' in line or 'error:' in line:
                detailed_error += line + '\n'
        for line in (e.stderr or '').split('\n'):
            if '[ERROR]' in line or 'error:' in line:
                detailed_error += line + '\n'
        
        result_item['correctness_info'] = f'Execution failed. Return code: {e.returncode}. Error: {detailed_error}'
    except subprocess.TimeoutExpired as e:
        print(f"  [FAIL] {op_name} execution timed out.")
        result_item['correctness_info'] = 'Timeout fault'
    except Exception as e:
        print(f"  [FAIL] {op_name} unknown error: {e}")
        result_item['correctness_info'] = f'Unknown error: {str(e)}'
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)
            
    return op_name, result_item

def main():
    print("============== 开始并行验证硬编码 Prompt 的测试 ==============\n")
    results = {}
    
    # 使用线程池并发运行
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(test_single_op, pf, op): op for pf, op in ENV_TEST_OPS}
        for future in as_completed(futures):
            op_name, res = future.result()
            results[op_name] = res

    # 按原顺序写入结果
    ordered_result = {}
    for _, op_name in ENV_TEST_OPS:
        if op_name in results:
            ordered_result[op_name] = results[op_name]

    print(f"\n[INFO] 将结果写入到 {output_file}")
    with open(output_file, 'w') as f:
        json.dump(ordered_result, f, indent=2, ensure_ascii=False)

    print("\n================ 总结 ================")
    for pf, op_name in ENV_TEST_OPS:
        res = ordered_result.get(op_name, {})
        if res.get("compiled") == False:
            print(f"{op_name}: 编译失败")
        else:
            print(f"{op_name}: 编译={'成功' if res.get('compiled') else '失败'}, 正确性={res.get('correctness')}")
    print("============== 验证结束 ==============")

if __name__ == '__main__':
    main()
