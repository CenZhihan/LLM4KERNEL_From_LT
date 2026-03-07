
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor hardtanh_impl_npu(const at::Tensor& self, double min_val, double max_val) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnHardtanhCustom, self, min_val, max_val, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("hardtanh_custom", &hardtanh_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hardtanh_custom", &hardtanh_impl_npu, "HardTanh activation");
}
