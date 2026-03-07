
#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    SoftmaxCustomTilingData tiling;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    int32_t axis = 1;
    if (attrs != nullptr) {
        const int32_t *axisPtr = attrs->GetAttrPointer<int32_t>(0);
        if (axisPtr != nullptr) {
            axis = *axisPtr;
        }
    }

    const gert::Shape *inShape = context->GetInputShape(0);
    auto origin = inShape->GetOriginShape();
    uint32_t dimNum = origin.GetDimNum();
    if (axis < 0) {
        axis += static_cast<int32_t>(dimNum);
    }
    uint32_t cols = origin.GetDim(static_cast<uint32_t>(axis));
    uint64_t totalLength = origin.GetShapeSize();
    uint32_t totalRows = static_cast<uint32_t>(totalLength / cols);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalRows(totalRows);
    tiling.set_cols(cols);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_axis(axis);

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
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name)
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

OP_ADD(SoftmaxCustom);
}
