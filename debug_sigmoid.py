import torch
import torch_npu
import sys
sys.path.append('ascend_op_projects/CppExtension_sigmoid_custom/build/lib.linux-aarch64-cpython-310')
import custom_ops_lib_sigmoid_custom

x = torch.abs(torch.rand(10, 10)) + 1e-2
x = x.npu()
y = custom_ops_lib_sigmoid_custom.sigmoid_custom(x)
print("X:", x.flatten()[:10])
print("Y:", y.flatten()[:10])
