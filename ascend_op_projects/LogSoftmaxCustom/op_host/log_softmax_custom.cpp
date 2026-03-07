
#include "log_softmax_custom_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>

namespace {
inline uint32_t CeilDiv(uint64_t a, uint64_t b) {
    return static_cast<uint32_t>((a + b - 1) / b);
}
}

namespace optiling {
static const uint32_t DEFAULT_BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    LogSoftmaxCustomTilingData tiling;

    const gert::Shape *in_shape = context->GetInputShape(0);
    int64_t rank = static_cast<int64_t>(in_shape->GetOriginShape().GetDimNum());
    int64_t dim = 1;
    if (const gert::RuntimeAttrs *attrs = context->GetAttrs()) {
        const int32_t *dim_ptr = attrs->GetAttrPointer<int32_t>(0);
        if (dim_ptr != nullptr) {
            dim = static_cast<int64_t>(*dim_ptr);
        }
    }
    if (dim < 0) dim += rank;
    if (dim < 0) dim = 0;
    if (dim >= rank) dim = rank - 1;

    uint64_t totalLength = in_shape->GetOriginShape().GetShapeSize();
    uint64_t reduceLen = static_cast<uint64_t>(in_shape->GetOriginShape().GetDim(static_cast<uint32_t>(dim)));
    if (reduceLen == 0) reduceLen = 1;
    uint64_t rows = totalLength / reduceLen;

    // Choose a reasonable tile along reduction dimension
    uint32_t tileCols = static_cast<uint32_t>(std::min<uint64_t>(reduceLen, 4096));

    // Block dim limited by rows
    uint32_t blockDim = static_cast<uint32_t>(std::min<uint64_t>(DEFAULT_BLOCK_DIM, rows == 0 ? 1ULL : rows));

    context->SetBlockDim(blockDim);

    tiling.set_rows(static_cast<uint32_t>(rows));
    tiling.set_cols(static_cast<uint32_t>(reduceLen));
    tiling.set_tileCols(tileCols);
    tiling.set_dim(static_cast<int32_t>(dim));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

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
        this->Attr("dim").AttrType(OPTIONAL).Int(1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSoftmaxCustom);
}
