
#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t MAX_BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SoftmaxCustomTilingData tiling;

    const gert::Shape* in_shape = context->GetInputShape(0);
    // Expecting 2D input: [N, D]; fall back to treat last dim as row
    uint32_t dims = in_shape->GetOriginShape().GetDimNum();
    uint32_t N = 1;
    uint32_t D = 1;
    if (dims >= 2) {
        N = static_cast<uint32_t>(in_shape->GetOriginShape().GetDim(0));
        D = static_cast<uint32_t>(in_shape->GetOriginShape().GetDim(1));
    } else if (dims == 1) {
        N = 1;
        D = static_cast<uint32_t>(in_shape->GetOriginShape().GetDim(0));
    } else {
        // Scalar not supported
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = (N < MAX_BLOCK_DIM) ? N : MAX_BLOCK_DIM;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    uint32_t rowsPerBlock = (N + blockDim - 1) / blockDim;

    tiling.set_totalRows(N);
    tiling.set_rowSize(D);
    tiling.set_rowsPerBlock(rowsPerBlock);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 0;

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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SoftmaxCustom);
}
