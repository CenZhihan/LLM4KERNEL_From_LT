#!/usr/bin/env python3
"""
AscendC 环境自检脚本：用多个「可编译算子」走完整流程，验证环境是否正常。

对每个算子执行：
  1. 编译：msopgen + build.sh + deploy + pybind
  2. 运行：在 NPU 上跑自定义算子，与 PyTorch 参考实现对比

若全部通过，说明环境、CANN、驱动、torch_npu 与流水线均正常，evaluation 中的错误可归因于 LLM；
若某一项失败，可根据报错排查环境或该算子的 prompt。
"""
import os
import sys

# 保证从项目根目录执行
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch


def _test_add(device, custom_ops_lib):
    n = 1024
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    c = custom_ops_lib.add_custom(a, b)
    expected = a + b
    if not torch.allclose(c.cpu(), expected.cpu(), rtol=1e-4, atol=1e-5):
        return False, "add_custom(a,b) != a+b"
    return True, "add_custom(a,b) == a+b"


def _test_leaky_relu(device, custom_ops_lib):
    negative_slope = 0.1
    n = 1024
    x = torch.randn(n, dtype=torch.float32, device=device)
    y = custom_ops_lib.leaky_relu_custom(x, negative_slope)
    expected = torch.nn.functional.leaky_relu(x.cpu(), negative_slope).to(device)
    if not torch.allclose(y.cpu(), expected.cpu(), rtol=1e-3, atol=1e-4):
        return False, "leaky_relu_custom(x, 0.1) != F.leaky_relu(x, 0.1)"
    return True, "leaky_relu_custom 与 F.leaky_relu 一致"


def _test_reduce_sum(device, custom_ops_lib):
    n = 1024
    x = torch.randn(n, dtype=torch.float32, device=device)
    out = custom_ops_lib.reduce_sum_custom(x)
    ref = x.sum()
    # 该 custom 输出可能为标量或 [32] 的部分和，统一按「与 x.sum() 数值一致」校验
    out_val = out.sum() if out.numel() > 1 else out
    if not torch.allclose(out_val.cpu(), ref.cpu(), rtol=1e-3, atol=1e-4):
        return False, "reduce_sum_custom(x) 与 x.sum() 不一致"
    return True, "reduce_sum_custom 与 x.sum() 一致"


# 配置：(prompt 文件名, 后端 op 名, 简短描述, 运行并校验函数)
ENV_TEST_OPS = [
    ("ascendc_new_model_reduce_sum.py", "reduce_sum", "ReduceSum", _test_reduce_sum),
]


def main():
    import torch
    import torch_npu
    from backends.ascendc_backend import AscendBackend

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    backend = AscendBackend()
    total = len(ENV_TEST_OPS)
    for idx, (prompt_file, op_name, desc, run_test) in enumerate(ENV_TEST_OPS, start=1):
        print(f"[{idx}/{total}] {desc}：加载源码并编译...")
        prompt_path = os.path.join(project_root, "prompts", prompt_file)
        if not os.path.exists(prompt_path):
            print(f"  [FAIL] 未找到 {prompt_path}")
            return 1
        with open(prompt_path, "r", encoding="utf-8") as f:
            generated_code = f.read()
        compiled, compile_info = backend.compile(generated_code, op_name)
        if not compiled:
            print(f"  [FAIL] 编译失败:\n{compile_info}")
            return 1
        print(f"        编译成功，运行并校验...")
        # 每次编译会覆盖 CppExtension 的 .so，必须从 sys.modules 移除再 import 才能加载新 .so（reload 对 native 模块常无效）
        if "custom_ops_lib" in sys.modules:
            del sys.modules["custom_ops_lib"]
        import custom_ops_lib
        try:
            ok, msg = run_test(device, custom_ops_lib)
            if not ok:
                print(f"  [FAIL] {msg}")
                return 1
            print(f"        通过 ({msg})")
        except Exception as e:
            print(f"  [FAIL] 运行出错: {e}")
            return 1
    backend.cleanup()
    print(f"[OK] 环境自检通过：共 {total} 个算子编译与运行均正常。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
