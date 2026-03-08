
#include "log_softmax_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>

namespace {
inline int64_t NormalizeAxis(int64_t axis, int64_t rank) {
    if (axis < 0) axis += rank;
    return axis;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogSoftmaxCustomTilingData tiling;

    const gert::Shape* in_shape = context->GetInputShape(0);
    auto origin = in_shape->GetOriginShape();

    int64_t rank = static_cast<int64_t>(origin.GetDimNum());
    int64_t axis = 1; // default
    if (context->GetAttrs() != nullptr) {
        axis = context->GetAttrs()->GetInt("axis", 1);
    }
    axis = NormalizeAxis(axis, rank);

    uint64_t total = origin.GetShapeSize();
    uint64_t reduce_len = static_cast<uint64_t>(origin.GetDim(axis));
    uint64_t rows = total / reduce_len; // flatten outer*inner, reduce along axis

    // Static configuration
    const uint32_t BLOCK_DIM = std::min<uint32_t>(static_cast<uint32_t>(rows), 32);
    const uint32_t TILE_SIZE = 4096; // elements per tile (must fit UB)

    uint32_t cols = static_cast<uint32_t>(reduce_len);
    uint32_t tileNum = (cols + TILE_SIZE - 1) / TILE_SIZE;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_rows(static_cast<uint32_t>(rows));
    tiling.set_cols(cols);
    tiling.set_tileSize(TILE_SIZE);
    tiling.set_tileNum(tileNum);
    tiling.set_axis(static_cast<int32_t>(axis));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class LogSoftmaxCustom : public OpDef {
public:
    explicit LogSoftmaxCustom(const char* name) : OpDef(name)
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

        this->Attr("axis").Int(1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSoftmaxCustom);
}
