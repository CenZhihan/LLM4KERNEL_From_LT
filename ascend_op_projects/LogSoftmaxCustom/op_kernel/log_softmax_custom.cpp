
#include "kernel_operator.h"
#define __NPU_TILING__
#include "log_softmax_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLogSoftmax {
public:
    __aicore__ inline KernelLogSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t rows, uint32_t cols,
                                uint32_t tileSize, uint32_t tileNum)
    {
        this->rows = rows;
        this->cols = cols;
        this->tileSize = tileSize;
        this->tileNum = tileNum;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, static_cast<uint64_t>(rows) * cols);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, static_cast<uint64_t>(rows) * cols);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileSize * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileSize * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpQueue, BUFFER_NUM, this->tileSize * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        for (uint32_t row = blockIdx; row < rows; row += blockNum) {
            Pass1ReduceMax(row);
            Pass2ReduceSum(row);
            Pass3Write(row);
        }
    }

private:
    __aicore__ inline void Pass1ReduceMax(uint32_t row)
    {
        float maxVal = -3.402823e+38f; // approx -FLT_MAX
        uint64_t base = static_cast<uint64_t>(row) * cols;

        for (uint32_t t = 0; t < tileNum; ++t) {
            uint32_t offset = t * tileSize;
            if (offset >= cols) break;
            uint32_t valid = AscendC::min(tileSize, cols - offset);

            AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
            AscendC::DataCopy(xLocal, xGm[base + offset], valid);

            float tileMax = AscendC::ReduceMax(xLocal, valid);
            maxVal = (tileMax > maxVal) ? tileMax : maxVal;

            inQueueX.FreeTensor(xLocal);
        }
        this->rowMax = maxVal;
    }

    __aicore__ inline void Pass2ReduceSum(uint32_t row)
    {
        uint64_t base = static_cast<uint64_t>(row) * cols;
        float sumExp = 0.0f;

        for (uint32_t t = 0; t < tileNum; ++t) {
            uint32_t offset = t * tileSize;
            if (offset >= cols) break;
            uint32_t valid = AscendC::min(tileSize, cols - offset);

            AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
            AscendC::LocalTensor<DTYPE_X> tmpLocal = tmpQueue.AllocTensor<DTYPE_X>();

            AscendC::DataCopy(xLocal, xGm[base + offset], valid);
            AscendC::Dup(tmpLocal, static_cast<DTYPE_X>(rowMax), valid);
            AscendC::Sub(xLocal, xLocal, tmpLocal, valid);
            AscendC::Exp(xLocal, xLocal, valid);

            float tileSum = AscendC::ReduceSum(xLocal, valid);
            sumExp += tileSum;

            tmpQueue.FreeTensor(tmpLocal);
            inQueueX.FreeTensor(xLocal);
        }
        this->rowLogSum = AscendC::LogScalar(sumExp);
    }

    __aicore__ inline void Pass3Write(uint32_t row)
    {
        uint64_t base = static_cast<uint64_t>(row) * cols;

        for (uint32_t t = 0; t < tileNum; ++t) {
            uint32_t offset = t * tileSize;
            if (offset >= cols) break;
            uint32_t valid = AscendC::min(tileSize, cols - offset);

            AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
            AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            AscendC::LocalTensor<DTYPE_X> tmpLocal = tmpQueue.AllocTensor<DTYPE_X>();

            AscendC::DataCopy(xLocal, xGm[base + offset], valid);
            AscendC::Dup(tmpLocal, static_cast<DTYPE_X>(rowMax), valid);
            AscendC::Sub(xLocal, xLocal, tmpLocal, valid); // x - max
            AscendC::Dup(tmpLocal, static_cast<DTYPE_X>(rowLogSum), valid);
            AscendC::Sub(yLocal, xLocal, tmpLocal, valid); // x - max - logSum

            AscendC::DataCopy(yGm[base + offset], yLocal, valid);

            tmpQueue.FreeTensor(tmpLocal);
            outQueueY.FreeTensor(yLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tmpQueue;

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t rows;
    uint32_t cols;
    uint32_t tileSize;
    uint32_t tileNum;

    float rowMax;
    float rowLogSum;
};

extern "C" __global__ __aicore__ void log_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLogSoftmax op;
    op.Init(x, y,
            tiling_data.rows,
            tiling_data.cols,
            tiling_data.tileSize,
            tiling_data.tileNum);
    op.Process();
}
