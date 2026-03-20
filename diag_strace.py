#!/usr/bin/env python3
"""
Minimal: set ASCEND_CUSTOM_OPP_PATH, import torch_npu, load relu, call it.
Run under strace to see what files the CANN runtime accesses.
"""
import os, sys

# Set env BEFORE any Ascend imports
opp_path = "/workspace/LLM4KERNEL_From_LT/ascend_op_projects/opp_relu_custom/vendors/customize"
os.environ["ASCEND_CUSTOM_OPP_PATH"] = opp_path
lib_dir = opp_path + "/op_api/lib/"
os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import torch_npu
import custom_ops_lib_relu_custom

device_id = int(os.environ.get("NPU_DEVICE", "0"))
device = torch.device(f"npu:{device_id}")
x = torch.randn(8, 8, dtype=torch.float32, device=device)
torch_npu.npu.synchronize(device=device)

y = custom_ops_lib_relu_custom.relu_custom(x)
torch_npu.npu.synchronize(device=device)

print(f'Input: {x.flatten()[:5].tolist()}')
print(f'Output: {y.flatten()[:5].tolist()}')
ref = torch.relu(x)
print(f'Expected: {ref.flatten()[:5].tolist()}')
print(f'Match: {torch.allclose(y, ref, atol=1e-4)}')
