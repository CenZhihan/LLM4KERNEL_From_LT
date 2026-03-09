# 这个文件的主要目的是生成4个简单的activation算子的kernel文件，用于验证G哥的算子。
import os

base_dir = "output/ascendc/rag_four_activation/0.0-1.0/gpt-5/run0"
os.makedirs(base_dir, exist_ok=True)

ops = {
    "relu": {
        "compute": """
        AscendC::LocalTensor<DTYPE_X> tempLocal = calcQueue.AllocTensor<DTYPE_X>();
        AscendC::Duplicate(tempLocal, static_cast<DTYPE_X>(0.0f), alignedLength);
        AscendC::Max(yLocal, xLocal, tempLocal, alignedLength);
        calcQueue.FreeTensor(tempLocal);
"""
    },
    "tanh": {
        "compute": """
        AscendC::LocalTensor<DTYPE_X> tempLocal = calcQueue.AllocTensor<DTYPE_X>();
        
        // y = x * 2
        AscendC::Duplicate(tempLocal, static_cast<DTYPE_X>(2.0f), alignedLength);
        AscendC::Mul(yLocal, xLocal, tempLocal, alignedLength);
        
        // y = exp(2x)
        AscendC::Exp(yLocal, yLocal, alignedLength);
        
        // tempLocal = 1
        AscendC::Duplicate(tempLocal, static_cast<DTYPE_X>(1.0f), alignedLength);
        
        // xLocal = exp(2x) + 1
        AscendC::Add(xLocal, yLocal, tempLocal, alignedLength);
        // xLocal = 1 / (exp(2x) + 1)
        AscendC::Reciprocal(xLocal, xLocal, alignedLength);
        
        // yLocal = exp(2x) - 1
        AscendC::Sub(yLocal, yLocal, tempLocal, alignedLength);
        
        // yLocal = yLocal * xLocal
        AscendC::Mul(yLocal, yLocal, xLocal, alignedLength);

        calcQueue.FreeTensor(tempLocal);
"""
    },
    "softplus": {
        "compute": """
        AscendC::LocalTensor<DTYPE_X> tempLocal = calcQueue.AllocTensor<DTYPE_X>();
        
        // y = exp(x)
        AscendC::Exp(yLocal, xLocal, alignedLength);
        
        // tempLocal = 1
        AscendC::Duplicate(tempLocal, static_cast<DTYPE_X>(1.0f), alignedLength);
        
        // y = exp(x) + 1
        AscendC::Add(yLocal, yLocal, tempLocal, alignedLength);
        
        // y = ln(1 + exp(x))
        AscendC::Ln(yLocal, yLocal, alignedLength);
        
        calcQueue.FreeTensor(tempLocal);
"""
    },
    "softsign": {
        "compute": """
        AscendC::LocalTensor<DTYPE_X> tempLocal = calcQueue.AllocTensor<DTYPE_X>();
        
        // yLocal = abs(x)
        AscendC::Abs(yLocal, xLocal, alignedLength);
        
        // tempLocal = 1
        AscendC::Duplicate(tempLocal, static_cast<DTYPE_X>(1.0f), alignedLength);
        
        // yLocal = abs(x) + 1
        AscendC::Add(yLocal, yLocal, tempLocal, alignedLength);
        
        // yLocal = 1 / (abs(x) + 1)
        AscendC::Reciprocal(yLocal, yLocal, alignedLength);
        
        // yLocal = x * yLocal
        AscendC::Mul(yLocal, xLocal, yLocal, alignedLength);

        calcQueue.FreeTensor(tempLocal);
"""
    }
}

template = """project_json_src='''
[
    {{
        "op": "{op_camel}Custom",
        "language": "cpp",
        "input_desc": [
            {{
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }}
        ],
        "output_desc": [
            {{
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }}
        ]
    }}
]
'''

host_tiling_src=\"\"\"
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF({op_camel}CustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, lastBlockLength);
  TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({op_camel}Custom, {op_camel}CustomTilingData)
}}
\"\"\"

host_operator_src=\"\"\"
#include "{op}_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {{
const uint32_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{{
    {op_camel}CustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    
    uint32_t ALIGN_NUM = 8;
    uint32_t core_num = 1; 

    uint32_t block_dim = core_num;
    uint32_t blockLength = totalLength; 
    uint32_t lastBlockLength = totalLength;

    uint32_t tileLength = 2048; 
    uint32_t tileNum = (blockLength + tileLength - 1) / tileLength;
    uint32_t lastTileLength = blockLength - (tileNum - 1) * tileLength;
    if (lastTileLength == 0) {{
        lastTileLength = tileLength;
    }}

    context->SetBlockDim(block_dim);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_block_dim(block_dim);
    tiling.set_blockLength(blockLength);
    tiling.set_tileLength(tileLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.set_lastTileLength(lastTileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ge {{
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{{
    const gert::Shape* in_shape = context->GetInputShape(0);
    gert::Shape* out_shape = context->GetOutputShape(0);
    *out_shape = *in_shape;
    return GRAPH_SUCCESS;
}}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}}
}}

namespace ops {{
class {op_camel}Custom : public OpDef {{
public:
    explicit {op_camel}Custom(const char* name) : OpDef(name)
    {{
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({{ge::DT_FLOAT}})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({{ge::FORMAT_ND}});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({{ge::DT_FLOAT}})
            .Format({{ge::FORMAT_ND}})
            .UnknownShapeFormat({{ge::FORMAT_ND}});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }}
}};

OP_ADD({op_camel}Custom);
}}
\"\"\"

kernel_src=\"\"\"
#include "kernel_operator.h"
#define __NPU_TILING__
#include "{op}_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class Kernel{op_camel} {{
public:
    __aicore__ inline Kernel{op_camel}() {{}}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_dim, uint32_t tileNum, uint32_t blockLength, uint32_t tileLength, uint32_t lastBlockLength, uint32_t lastTileLength)
    {{
        this->totalLength = totalLength;
        this->ALIGN_NUM = ALIGN_NUM;
        this->block_dim = block_dim;
        this->tileNum = tileNum;
        this->blockLength = blockLength;
        this->tileLength = tileLength;
        this->lastBlockLength = lastBlockLength;
        this->lastTileLength = lastTileLength;
        
        uint32_t allocLength = (this->tileLength + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        uint32_t alignedBlockLength = (this->blockLength + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, alignedBlockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, alignedBlockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, allocLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, allocLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(calcQueue, 1, allocLength * sizeof(DTYPE_X));
    }}

    __aicore__ inline void Process()
    {{
        for (int32_t i = 0; i < this->tileNum; i++) {{
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {{
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        uint32_t currentLength = this->tileLength;
        if (progress == this->tileNum - 1) {{
            currentLength = this->lastTileLength;
        }}
        
        uint32_t alignedLength = (currentLength + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], alignedLength);
        inQueueX.EnQue(xLocal);
    }}

    __aicore__ inline void Compute(int32_t progress)
    {{
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        
        uint32_t currentLength = this->tileLength;
        if (progress == this->tileNum - 1) {{
            currentLength = this->lastTileLength;
        }}
        
        uint32_t alignedLength = (currentLength + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;

{compute_logic}

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }}

    __aicore__ inline void CopyOut(int32_t progress)
    {{
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        uint32_t currentLength = this->tileLength;
        if (progress == this->tileNum - 1) {{
            currentLength = this->lastTileLength;
        }}
        
        uint32_t alignedLength = (currentLength + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, alignedLength);
        outQueueY.FreeTensor(yLocal);
    }}

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> calcQueue;

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t ALIGN_NUM;
    uint32_t block_dim;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t lastBlockLength;
    uint32_t lastTileLength;
}};

extern "C" __global__ __aicore__ void {op}_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tiling_data, tiling);
    Kernel{op_camel} op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_dim, tiling_data.tileNum, tiling_data.blockLength, tiling_data.tileLength, tiling_data.lastBlockLength, tiling_data.lastTileLength);
    op.Process();
}}
\"\"\"

python_bind_src=\"\"\"
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor {op}_custom_impl_npu(const at::Tensor& self) {{
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnn{op_camel}Custom, self, result);
    return result;
}}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {{
    m.impl("{op}_custom", &{op}_custom_impl_npu);
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{op}_custom", &{op}_custom_impl_npu, "{op}(x)");
}}
\"\"\"

model_src='''
import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.{op}_custom(x)
'''
"""

for op, data in ops.items():
    op_camel = "".join(word.capitalize() for word in op.split("_"))
    content = template.format(op=op, op_camel=op_camel, compute_logic=data["compute"])
    with open(os.path.join(base_dir, f"{op}.txt"), "w") as f:
        f.write(content)
