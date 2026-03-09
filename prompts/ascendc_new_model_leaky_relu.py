project_json_src='''
[
    {
        "op": "LeakyReluCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ],
        "attr": [
            {
                "name": "negative_slope",
                "param_type": "optional",
                "type": "float",
                "default_value": "0.1"
            }
        ]
    }
]

'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LeakyReluCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, lastBlockLength);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(float, negativeSlope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LeakyReluCustom, LeakyReluCustomTilingData)
}
"""


host_operator_src="""
#include "leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 48;
const uint32_t ALIGN_NUM = 8;
const uint32_t MAX_TILE_LENGTH = 8192; 

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    LeakyReluCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *negativeSlope = attrs->GetAttrPointer<float>(0);
    
    uint32_t blockLength = totalLength / BLOCK_DIM;
    blockLength = (blockLength / ALIGN_NUM) * ALIGN_NUM;
    uint32_t lastBlockLength = totalLength - blockLength * (BLOCK_DIM - 1);
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.set_tileLength(MAX_TILE_LENGTH);
    tiling.set_negativeSlope(*negativeSlope);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x1_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x1_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class LeakyReluCustom : public OpDef {
public:
    explicit LeakyReluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.1);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(LeakyReluCustom);
}
"""


kernel_src="""
#include "kernel_operator.h"
#define __NPU_TILING__
#include "leaky_relu_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelLeakyRelu {
public:
    __aicore__ inline KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t blockLength, uint32_t lastBlockLength, uint32_t tileLength, float negativeSlope)
    {
        this->negativeSlope = static_cast<float>(negativeSlope);
        this->tileLength = tileLength;
        
        if (AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1) {
            this->blockLength = lastBlockLength;
        } else {
            this->blockLength = blockLength;
        }
        
        if (this->blockLength == 0) return;
        
        this->tileNum = this->blockLength / this->tileLength;
        this->tailLength = this->blockLength % this->tileLength;

        uint32_t offset = blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ float *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + offset, this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        if (this->tailLength > 0) {
            CopyIn(this->tileNum, this->tailLength);
            Compute(this->tileNum, this->tailLength);
            CopyOut(this->tileNum, this->tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpTensor1 = tmpBuffer1.Get<float>();
        AscendC::LocalTensor<float> tmpTensor2 = tmpBuffer2.Get<float>();
        float inputVal = 0.0;
        
        // Pad length to multiple of 8 (or 16 bytes for float, 32 bytes for datamove) for computation if needed
        // but Maxs/Mins support length
        AscendC::Maxs(tmpTensor1, xLocal, inputVal, length);
        AscendC::Mins(tmpTensor2, xLocal, inputVal, length);
        AscendC::Muls(tmpTensor2, tmpTensor2, this->negativeSlope, length);
        AscendC::Add(yLocal, tmpTensor1, tmpTensor2, length);
        
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tailLength;
    uint32_t tileLength;
    float negativeSlope;
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tiling_data.blockLength, tiling_data.lastBlockLength, tiling_data.tileLength, tiling_data.negativeSlope);
    op.Process();
}
"""


python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor leaky_relu_impl_npu(const at::Tensor& self, double negative_slope) {
    // float argument not supported now, so use double negative_slope
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnLeakyReluCustom, self, negative_slope, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("leaky_relu_custom", &leaky_relu_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_custom", &leaky_relu_impl_npu, "LeakyReLU activation");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib
class ModelNew(torch.nn.Module):
    def __init__(self, negative_slope: float = 0.1):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.leaky_relu_custom(x, self.negative_slope)
'''
