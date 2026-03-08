
#include "kernel_operator.h"
#define __NPU_TILING__
#include "softmax_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalRows, uint32_t rowSize, uint32_t rowsPerBlock) {
        this->totalRows = totalRows;
        this->rowSize = rowSize;
        this->rowsPerBlock = rowsPerBlock;

        uint32_t baseRow = AscendC::GetBlockIdx() * rowsPerBlock;
        uint32_t baseOffset = baseRow * rowSize;

        xGm.SetGlobalBuffer((__gm__ float*)x + baseOffset, rowsPerBlock * rowSize);
        yGm.SetGlobalBuffer((__gm__ float*)y + baseOffset, rowsPerBlock * rowSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, rowSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        // Pipeline by rows
        int32_t loopCount = rowsPerBlock;
        // CopyIn and Compute/CopyOut with double-buffering
        for (int32_t i = 0; i < loopCount; ++i) {
            if (!RowIsValid(i)) {
                break;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline bool RowIsValid(int32_t localRow) const {
        uint32_t globalRow = AscendC::GetBlockIdx() * rowsPerBlock + static_cast<uint32_t>(localRow);
        return globalRow < totalRows;
    }

    __aicore__ inline void CopyIn(int32_t localRow) {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[static_cast<uint32_t>(localRow) * rowSize], rowSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t) {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Compute softmax along the row: y = softmax(x)
        // Assume AscendC provides a fused Softmax; otherwise, backend may lower to appropriate sequence.
        AscendC::Softmax(yLocal, xLocal, rowSize);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t localRow) {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[static_cast<uint32_t>(localRow) * rowSize], yLocal, rowSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t totalRows;
    uint32_t rowSize;
    uint32_t rowsPerBlock;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.rowSize, tiling_data.rowsPerBlock);
    op.Process();
}
