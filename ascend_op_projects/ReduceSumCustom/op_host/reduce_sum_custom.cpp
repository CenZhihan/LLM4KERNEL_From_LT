
#include "reduce_sum_custom_tiling.h"
#include "register/op_def_registry.h"
#define REDUCE_TILING_0 1
#define REDUCE_TILING_1 2
#define REDUCE_TILING_2 3

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t ONE_REPEAT_LEN = 256;
constexpr uint32_t ONE_BLOCK_LEN = 32;
constexpr uint32_t OUT_SHAPE = 32;
constexpr uint32_t FLOAT_THRESHOLD0 = ONE_REPEAT_LEN / sizeof(float);
constexpr uint32_t FLOAT_THRESHOLD1 = ONE_REPEAT_LEN / sizeof(float) * ONE_BLOCK_LEN / sizeof(float);
constexpr uint32_t FLOAT_THRESHOLD2 = ONE_REPEAT_LEN / sizeof(float) * ONE_REPEAT_LEN / sizeof(float);
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto inputDtype = context->GetInputTensor(0)->GetDataType();
    // Only WholeReduceSum is used under 256B.
    if (totalLength <= FLOAT_THRESHOLD0 && inputDtype == ge::DT_FLOAT) {
        context->SetTilingKey(REDUCE_TILING_0);
    // One WholeReduceSum and one BlockReduceSum are used in (256B,2KB](for float input).
    } else if (totalLength <= FLOAT_THRESHOLD1 && inputDtype == ge::DT_FLOAT) {
        context->SetTilingKey(REDUCE_TILING_1);
    // Two WholeReduceSum are used in (2KB,16KB](for float input).
    } else if (totalLength <= FLOAT_THRESHOLD2 && inputDtype == ge::DT_FLOAT) {
        context->SetTilingKey(REDUCE_TILING_2);
    }
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_outLength(OUT_SHAPE);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = {optiling::OUT_SHAPE};
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ReduceSumCustom : public OpDef {
public:
    explicit ReduceSumCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(ReduceSumCustom);
} // namespace ops
