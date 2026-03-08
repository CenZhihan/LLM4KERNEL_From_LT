
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor conv2d_add_scale_sigmoid_group_norm_custom_impl_npu(const at::Tensor& x,
                                                               const at::Tensor& bias,
                                                               const at::Tensor& scale,
                                                               const at::Tensor& gamma,
                                                               const at::Tensor& beta) {
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnConv2dAddScaleSigmoidGroupNormCustom, x, bias, scale, gamma, beta, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_add_scale_sigmoid_group_norm_custom", &conv2d_add_scale_sigmoid_group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_add_scale_sigmoid_group_norm_custom", &conv2d_add_scale_sigmoid_group_norm_custom_impl_npu, "conv2d add+scale+sigmoid+groupnorm (fused, custom)");
}
