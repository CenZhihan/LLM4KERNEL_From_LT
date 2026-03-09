
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor clamp_broadcast_custom_impl_npu(const at::Tensor& x, const at::Tensor& min_val, const at::Tensor& max_val) {
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnClampBroadcastCustom, x, min_val, max_val, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("clamp_broadcast_custom", &clamp_broadcast_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("clamp_broadcast_custom", &clamp_broadcast_custom_impl_npu, "clamp with per-channel min/max");
}
