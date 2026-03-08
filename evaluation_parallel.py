# Parallel version: evaluation with ThreadPoolExecutor
import os
import json
import subprocess
import tempfile
from dataset import dataset
from config import temperature, top_p
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

EVAL_TIMEOUT = 360


def eval_single_op(op, out_dir, language):
    """
    Run evaluation for one op. Returns (op, result_dict).
    result_dict has keys: compiled, correctness, performance, compile_info/correctness_info as in evaluation.py.
    """
    result = {'compiled': False, 'correctness': None, 'performance': None}
    op_path = os.path.join(out_dir, f'{op}.txt')
    if not os.path.exists(op_path):
        print(f"[FAIL] op {op}: missing {op_path}")
        result['correctness_info'] = 'Missing generated file'
        return op, result
    try:
        with open(op_path, 'r') as saved_log:
            response_txt = saved_log.read()
    except Exception as e:
        result['correctness_info'] = str(e)
        return op, result

    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.txt') as tf_input, \
            tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tf_out:
        tf_input.write(response_txt)
        tf_input.flush()
        out_path = tf_out.name
        try:
            try:
                captured_text = subprocess.run(
                    ['python3', 'eval_single_runner.py', tf_input.name, op, language, out_path],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=EVAL_TIMEOUT,
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
                print(f'[INFO] Evaluating op {op} -> {result_item}')
                return op, result_item
            except subprocess.CalledProcessError as e:
                if 'FileNotFoundError' in (e.stderr or ''):
                    print(f"[FAIL] op {op}: FileNotFoundError - check project_root_path in config.py")
                    result['correctness_info'] = 'FileNotFoundError'
                elif e.returncode == -11:
                    print(f"[FAIL] op {op}: Segmentation fault")
                    result = {'compiled': True, 'correctness': None, 'performance': None, 'correctness_info': 'Segmentation fault'}
                else:
                    print(f"[FAIL] op {op}: {e.stderr}")
                    result = {'compiled': True, 'correctness': None, 'performance': None, 'correctness_info': 'Unknown fault'}
                return op, result
            except subprocess.TimeoutExpired:
                print(f"[FAIL] op {op}: run timeout")
                result = {'compiled': True, 'correctness': None, 'performance': None, 'correctness_info': 'Timeout fault'}
                return op, result
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation in parallel.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--model', type=str, default='deepseek-chat', help='Model name')
    parser.add_argument('--language', type=str, default='cuda', help='Programming language')
    parser.add_argument('--strategy', type=str, default='add_shot', help='Strategy type.')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='List of categories.')
    parser.add_argument('--skip_existing', action='store_true', help='Skip when result_*.json already exists.')
    parser.add_argument('--workers', type=int, default=4, help='Max concurrent evaluations (default 4).')

    args = parser.parse_args()
    runs = args.runs
    model = args.model
    language = args.language
    strategy = args.strategy
    categories = args.categories
    skip_existing = args.skip_existing
    workers = args.workers

    print(f"Runs: {runs}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Strategy: {strategy}")
    print(f"Categories: {categories}")
    print(f"Workers: {workers}")

    op_tested = list(dataset.keys())
    if categories != ['all']:
        op_tested = [op for op in op_tested if dataset[op]['category'] in categories]

    if '/' in model:
        model_name = model.split('/')[1]
    else:
        model_name = model

    for run in range(runs):
        out_dir = f'output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}'
        if categories == ['all']:
            output_file = os.path.join(out_dir, 'result.json')
        else:
            output_file = os.path.join(out_dir, f'result_{"_".join(categories)}.json')
        if os.path.exists(output_file) and skip_existing:
            print(f"[INFO] Already evaluated, please see {output_file} (skip_existing=True)")
            continue
        if os.path.exists(output_file):
            print(f"[INFO] Result file exists and will be overwritten: {output_file}")

        result = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(eval_single_op, op, out_dir, language): op for op in op_tested}
            for future in as_completed(futures):
                op, result_item = future.result()
                result[op] = result_item

        # Preserve order of op_tested in output
        ordered_result = {op: result[op] for op in op_tested if op in result}
        with open(output_file, 'w') as f:
            print(f"[INFO] Evaluated successfully, write into {output_file}")
            json.dump(ordered_result, f, indent=2)
