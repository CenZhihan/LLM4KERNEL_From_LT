import sys
import os
import json
import importlib
import importlib.util
import traceback

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch_npu
from backends.ascendc_backend import AscendBackend

def test_matmul_add(device, custom_ops_lib):
    A = torch.randn(128, 128, dtype=torch.float32, device=device)
    B = torch.randn(128, 128, dtype=torch.float32, device=device)
    bias = torch.randn(128, dtype=torch.float32, device=device)
    out = custom_ops_lib.matmul_add_custom(A, B, bias)
    expected = torch.matmul(A, B) + bias
    if not torch.allclose(out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
        max_diff = torch.max(torch.abs(out.cpu() - expected.cpu()))
        detail = f"\n---> Max difference: {max_diff.item():.6f}\n" \
                 f"---> Ref  (first 10): {expected.flatten().cpu()[:10].tolist()}\n" \
                 f"---> New  (first 10): {out.flatten().cpu()[:10].tolist()}\n"
        return False, f"matmul_add_custom(A, B, bias) != A@B + bias {detail}"
    return True, "matmul_add_custom 结果正确"

def test_leaky_relu(device, custom_ops_lib):
    x = torch.randn(48, 8192, dtype=torch.float32, device=device)
    negative_slope = 0.1
    out = custom_ops_lib.leaky_relu_custom(x, negative_slope)
    expected = torch.nn.functional.leaky_relu(x.cpu(), negative_slope).to(device)
    if not torch.allclose(out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
        max_diff = torch.max(torch.abs(out.cpu() - expected.cpu()))
        detail = f"\n---> Max difference: {max_diff.item():.6f}\n" \
                 f"---> Ref  (first 10): {expected.flatten().cpu()[:10].tolist()}\n" \
                 f"---> New  (first 10): {out.flatten().cpu()[:10].tolist()}\n"
        return False, f"leaky_relu_custom 结果不一致 {detail}"
    return True, "leaky_relu_custom 结果正确"

def test_mse_loss(device, custom_ops_lib):
    x = torch.randn(32768, dtype=torch.float32, device=device)
    y = torch.randn(32768, dtype=torch.float32, device=device)
    out = custom_ops_lib.mse_loss_custom(x, y)
    expected = torch.nn.functional.mse_loss(x.cpu(), y.cpu()).to(device)
    if not torch.allclose(out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
        max_diff = torch.max(torch.abs(out.cpu() - expected.cpu()))
        detail = f"\n---> Max difference: {max_diff.item():.6f}\n" \
                 f"---> Ref  (first 10): {expected.flatten().cpu()[:10].tolist()}\n" \
                 f"---> New  (first 10): {out.flatten().cpu()[:10].tolist()}\n"
        return False, f"mse_loss_custom 结果不一致 {detail}"
    return True, "mse_loss_custom 结果正确"

def test_layer_norm(device, custom_ops_lib):
    x = torch.randn(864, 1024, dtype=torch.float32, device=device)
    gamma = torch.ones(1024, dtype=torch.float32, device=device)
    beta = torch.zeros(1024, dtype=torch.float32, device=device)
    epsilon = 1e-5
    out = custom_ops_lib.layer_norm_custom(x, gamma, beta, epsilon)
    expected = torch.nn.functional.layer_norm(x.cpu(), (1024,), weight=gamma.cpu(), bias=beta.cpu(), eps=epsilon).to(device)
    if not torch.allclose(out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3):
        max_diff = torch.max(torch.abs(out.cpu() - expected.cpu()))
        detail = f"\n---> Max difference: {max_diff.item():.6f}\n" \
                 f"---> Ref  (first 10): {expected.flatten().cpu()[:10].tolist()}\n" \
                 f"---> New  (first 10): {out.flatten().cpu()[:10].tolist()}\n"
        return False, f"layer_norm_custom 结果不一致 {detail}"
    return True, "layer_norm_custom 结果正确"

def test_reduce_sum(device, custom_ops_lib):
    x = torch.randn(256, dtype=torch.float32, device=device)
    out = custom_ops_lib.reduce_sum_custom(x)
    ref = x.sum()
    out_val = out.sum() if out.numel() > 1 else out
    if not torch.allclose(out_val.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3):
        max_diff = torch.max(torch.abs(out_val.cpu() - ref.cpu()))
        detail = f"\n---> Max difference: {max_diff.item():.6f}\n" \
                 f"---> Ref  (first 10): {ref.flatten().cpu()[:10].tolist()}\n" \
                 f"---> New  (first 10): {out_val.flatten().cpu()[:10].tolist()}\n"
        return False, f"reduce_sum_custom 结果不一致 {detail}"
    return True, "reduce_sum_custom 结果正确"

tests = {
    "matmul_add": test_matmul_add,
    "leaky_relu": test_leaky_relu,
    "mse_loss": test_mse_loss,
    "layer_norm": test_layer_norm,
    "reduce_sum": test_reduce_sum
}

def run():
    prompt_file = sys.argv[1]
    op_name = sys.argv[2]
    out_path = sys.argv[3]
    
    result = {'compiled': False, 'correctness': None, 'correctness_info': None}
    try:
        device = torch.device("npu:6")
        torch.npu.set_device(device)
        backend = AscendBackend()
        
        prompt_path = os.path.join(project_root, "prompts", prompt_file)
        if not os.path.exists(prompt_path):
            result['correctness_info'] = f"未找到 {prompt_path}"
            return result
            
        with open(prompt_path, "r", encoding="utf-8") as f:
            generated_code = f.read()
            
        compiled, compile_info = backend.compile(generated_code, op_name)
        if not compiled:
            result['compile_info'] = compile_info
            return result
        
        result['compiled'] = True
        
        module_name = f"custom_ops_lib_{op_name}_custom"
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # NOTE: dynamically load the .so file which is placed by ascend_compile in a custom directory
        # The deploy path was added to LD_LIBRARY_PATH but the Python module itself might only be available in standard paths
        # Actually it was pip installed in the compile step. Let's try to reload standard paths.
        import site
        importlib.reload(site)
        
        try:
            custom_ops_lib = importlib.import_module(module_name)
        except ModuleNotFoundError:
            # Maybe the wheel installed a different name or to a path not yet in sys.path
            import glob
            # Try to directly load the extension if import fails
            cpp_ext_dir = os.path.join(project_root, "ascend_op_projects", f"CppExtension_{op_name}_custom")
            wheel_dir = os.path.join(cpp_ext_dir, "dist")
            # If we need to directly find the .so:
            so_files = glob.glob(os.path.join(cpp_ext_dir, "build", "lib*", "*.so"))
            if so_files:
                spec = importlib.util.spec_from_file_location(module_name, so_files[0])
                custom_ops_lib = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = custom_ops_lib
                spec.loader.exec_module(custom_ops_lib)
            else:
                raise
        ok, msg = tests[op_name](device, custom_ops_lib)
        result['correctness'] = ok
        result['correctness_info'] = msg
        return result
    except Exception as e:
        err_msg = traceback.format_exc()
        result['correctness_info'] = f"运行出错: {e}\n{err_msg}"
        return result
    finally:
        with open(out_path, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    run()
