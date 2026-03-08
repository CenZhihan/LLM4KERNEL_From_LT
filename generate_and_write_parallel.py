# Parallel version: LLM kernel generation with ThreadPoolExecutor
from generate_and_write import generate_prompt, generate_and_write_single
from utils.utils import get_client, get_default_model_from_config
from config import temperature, top_p
from dataset import dataset
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def _generate_one(op, out_dir, language, strategy, model):
    """Single-op task: skip if exists, else generate and write. Returns (op, None) on success, (op, exception) on error."""
    out_path = os.path.join(out_dir, f'{op}.txt')
    if os.path.exists(out_path):
        print(f'[INFO] Already generated at {out_dir}/{op}.txt, skip')
        return op, None
    try:
        print(f'[INFO] Generate kernel for op {op}, strategy is {strategy}')
        prompt = generate_prompt(language, strategy, op)
        client = get_client(model)
        generate_and_write_single(prompt, client, out_dir, op, model)
        print(f'[INFO] Done op {op}')
        return op, None
    except Exception as e:
        print(f'[FAIL] op {op}: {e}')
        return op, e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model evaluation with specified parameters (parallel).")
    parser.add_argument('--runs', type=int, default=1, help='Number of runs.')
    _default_model = get_default_model_from_config() or 'deepseek-chat'
    parser.add_argument('--model', type=str, default=_default_model, help='Model name.')
    parser.add_argument('--language', type=str, default='cuda', help='Language to use.')
    parser.add_argument('--strategy', type=str, default='add_shot', help='Strategy type.')
    parser.add_argument('--categories', nargs='+', default=['activation'], help='List of categories.')
    parser.add_argument('--workers', type=int, default=4, help='Max concurrent API calls (default 4).')

    args = parser.parse_args()
    runs = args.runs
    model = args.model
    language = args.language
    strategy = args.strategy
    categories = args.categories
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
        os.makedirs(out_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_generate_one, op, out_dir, language, strategy, model): op
                for op in op_tested
            }
            for future in as_completed(futures):
                op, err = future.result()
                if err is not None:
                    pass  # already printed in _generate_one
