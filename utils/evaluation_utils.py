import re, os, json
from utils.utils import get_ref_src_path
from backends.backend_registry import BACKEND_REGISTRY
import torch
import importlib
import numpy as np
from dataset import dataset
from config import temperature, top_p, num_perf_trials

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output. Prefer blocks with explicit
    language tag (```python / ```cpp) so that leading description blocks
    (e.g. "、、、" or text in ``` without tag) from none-strategy outputs
    are not mistaken for code.
    """
    trimmed = output_string.strip()

    # 1) Prefer explicit ```python or ```cpp block (avoids taking a "description" block first)
    for code_type in code_language_types:
        pattern = rf"```{re.escape(code_type)}\s*\n(.*?)```"
        match = re.search(pattern, trimmed, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 2) Fallback: first generic ```...``` and strip optional language from first line
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)
    if code_match:
        code_block = code_match.group(1).strip()
        for code_type in code_language_types:
            if code_block.startswith(code_type):
                code_block = code_block[len(code_type) :].strip()
        return code_block

    return None


def _normalize_fullwidth_in_code(code: str) -> str:
    """Replace common fullwidth punctuation with ASCII so Python compile does not raise invalid character (e.g. model wrote Chinese in docstrings)."""
    if not code:
        return code
    replacements = [
        ("\uff08", "("),   # fullwidth (
        ("\uff09", ")"),   # fullwidth )
        ("\uff0c", ","),   # fullwidth ,
        ("\u3002", "."),   # Chinese period
        ("\uff1b", ";"),   # fullwidth ;
        ("\uff1a", ":"),   # fullwidth :
    ]
    for full, half in replacements:
        code = code.replace(full, half)
    return code


def eval_single(response_txt:str, op, language):
    # Try to dynamically import the backend if it's not yet registered
    if language not in BACKEND_REGISTRY:
        try:
            importlib.import_module(f"backends.{language}_backend")
        except ImportError as e:
            raise ValueError(f"Unsupported language/platform: {language} (module not found)") from e
    backend = BACKEND_REGISTRY.get(language)
    if backend is None:
        raise ValueError(f"Unsupported language/platform: {language}")
    
    hardware = backend.get_hardware_name()
    result = {'compiled': False, 'correctness': None, 'performance': None, 'hardware': hardware}
    try:
        generated_code = extract_first_code(response_txt, ['python', 'cpp'])
        if generated_code is None:
            generated_code = response_txt
        generated_code = _normalize_fullwidth_in_code(generated_code)
        compiled, compile_info = backend.compile(generated_code, op)
        if not compiled:
            result['compile_info'] = compile_info
            return result
        result['compiled'] = True
        ref_src_path = get_ref_src_path(op)
        with open(ref_src_path, 'r') as f:
            ref_src = f.read()
        correctness, info = backend.correctness_execution(ref_src)
        if not correctness:
            result['correctness_info'] = info
            return result
        result['correctness'] = True
        elapsed_times = backend.time_execution()
        result['performance'] = {
            "mean": float(f"{np.mean(elapsed_times):.3g}"),
            "std": float(f"{np.std(elapsed_times):.3g}"),
            "min": float(f"{np.min(elapsed_times):.3g}"),
            "max": float(f"{np.max(elapsed_times):.3g}"),
            "num_trials": len(elapsed_times),
        }
        backend.cleanup()
        return result
    finally:
        # ascendc：用完后删除该 op 的工程目录和 JSON，避免 ascend_op_projects 越积越多
        if hasattr(backend, 'cleanup_project_if_any'):
            backend.cleanup_project_if_any()

def eval_all(out_dir, language, op_tested=dataset.keys()):
    result = {}
    
    for op in op_tested:
        print(f"[INFO] eval op {op}")
        with open(os.path.join(out_dir, f'{op}.txt'), 'r') as saved_log:
            response_txt = saved_log.read()
        result[op] = eval_single(response_txt, op, language)
        
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    
if __name__ == '__main__':
    runs = 1
    model = 'deepseek-chat'
    language = 'cuda'
    op_tested = list(dataset.keys())
    op_tested = ['ltsm_hn', 'conv3d_leaky_relu_sum_clamp_gelu','square_matrix_multiplication','l2_norm','adam','sgd']
    select_shot = False
    for run in range(runs):
        if not select_shot:
            out_dir = f'output/{language}/add_shot/{temperature}-{top_p}/{model}/run{run}'
        else:
            out_dir = f'output/{language}/selected_shot/{temperature}-{top_p}/{model}/run{run}'
        eval_all(out_dir, language, op_tested)
