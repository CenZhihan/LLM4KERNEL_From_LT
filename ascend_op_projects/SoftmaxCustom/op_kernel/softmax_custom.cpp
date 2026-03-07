
#include "kernel_operator.h"
#define __NPU_TILING__
#include "softmax_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // buffering for streaming

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalRows, uint32_t cols,
                                uint32_t tileNum, int32_t axis)
    {
        this->cols = cols;
        this->tileNum = tileNum;
        this->axis = axis;

        // compute row partition for current block (load-balance tail)
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t baseRows = totalRows / blockNum;
        uint32_t tail = totalRows % blockNum;
        this->rowStart = blockIdx * baseRows + (blockIdx < tail ? blockIdx : tail);
        this->rowCount = baseRows + (blockIdx < tail ? 1 : 0);

        // tile length for streaming across columns
        uint32_t perLoop = tileNum * BUFFER_NUM;
        this->tileLength = (cols + perLoop - 1) / perLoop;

        // point GM buffers to this block's region
        xGm.SetGlobalBuffer((__gm__ float *)x + this->rowStart * this->cols, this->rowCount * this->cols);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->rowStart * this->cols, this->rowCount * this->cols);

        // buffers for one tile
        pipe.InitBuffer(tmpTileBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpTileBuf2, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // process row by row for numerical stability
        for (uint32_t r = 0; r < this->rowCount; ++r) {
            float rowMax = -3.402823466e+38F; // -FLT_MAX

            // pass 1: compute row-wise max across tiles
            for (uint32_t tileBase = 0; tileBase < this->cols; tileBase += this->tileLength) {
                uint32_t copyLen = this->cols - tileBase;
                if (copyLen > this->tileLength) copyLen = this->tileLength;

                AscendC::LocalTensor<float> xLocal = tmpTileBuf.Get<float>();
                AscendC::DataCopy(xLocal, xGm[r * this->cols + tileBase], copyLen);

                float tileMax = -3.402823466e+38F;
                // ReduceMax: compute max of the local tile
                AscendC::ReduceMax(tileMax, xLocal, copyLen);

                if (tileMax > rowMax) rowMax = tileMax;
            }

            // pass 2: compute row-wise sum of exp(x - max)
            float rowSum = 0.0f;
            for (uint32_t tileBase = 0; tileBase < this->cols; tileBase += this->tileLength) {
                uint32_t copyLen = this->cols - tileBase;
                if (copyLen > this->tileLength) copyLen = this->tileLength;

                AscendC::LocalTensor<float> xLocal = tmpTileBuf.Get<float>();
                AscendC::LocalTensor<float> expLocal = tmpTileBuf2.Get<float>();

                AscendC::DataCopy(xLocal, xGm[r * this->cols + tileBase], copyLen);
                // exp(x - rowMax)
                AscendC::Adds(expLocal, xLocal, -rowMax, copyLen);
                AscendC::Exp(expLocal, expLocal, copyLen);

                float tileSum = 0.0f;
                AscendC::ReduceSum(tileSum, expLocal, copyLen);
                rowSum += tileSum;
            }

            float invRowSum = 1.0f / rowSum;

            // pass 3: write normalized probabilities
            for (uint32_t tileBase = 0; tileBase < this->cols; tileBase += this->tileLength) {
                uint32_t copyLen = this->cols - tileBase;
                if (copyLen > this->tileLength) copyLen = this->tileLength;

                AscendC::LocalTensor<float> xLocal = tmpTileBuf.Get<float>();
                AscendC::LocalTensor<float> yLocal = tmpTileBuf2.Get<float>();

                AscendC::DataCopy(xLocal, xGm[r * this->cols + tileBase], copyLen);
                AscendC::Adds(yLocal, xLocal, -rowMax, copyLen);
                AscendC::Exp(yLocal, yLocal, copyLen);
                AscendC::Muls(yLocal, yLocal, invRowSum, copyLen);

                AscendC::DataCopy(yGm[r * this->cols + tileBase], yLocal, copyLen);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpTileBuf, tmpTileBuf2;

    AscendC::GlobalTensor<float> xGm, yGm;

    uint32_t cols;
    uint32_t tileNum;
    uint32_t tileLength;
    int32_t axis;

    uint32_t rowStart;
    uint32_t rowCount;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.cols, tiling_data.tileNum, tiling_data.axis);
    op.Process();
}
