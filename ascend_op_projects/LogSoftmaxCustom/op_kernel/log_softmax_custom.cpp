
#include "kernel_operator.h"
#define __NPU_TILING__
#include "log_softmax_custom_tiling_data.h"
#include <cfloat>

class KernelLogSoftmax {
public:
    __aicore__ inline KernelLogSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t rows, uint32_t cols, uint32_t tileCols, int32_t dim)
    {
        this->rows = rows;
        this->cols = cols;
        this->tileCols = tileCols;
        this->dim = dim;

        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(rows) * cols);
        yGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(rows) * cols);

        // local buffers
        pipe.InitBuffer(bufX, this->tileCols * sizeof(float));
        pipe.InitBuffer(bufY, this->tileCols * sizeof(float));
        // reduction temporaries
        pipe.InitBuffer(bufRed1, 256); // small buffer for reduction scalars
        pipe.InitBuffer(bufRed2, 256);
    }

    __aicore__ inline void Process()
    {
        // split rows across blocks
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t rowStart = static_cast<uint32_t>((static_cast<uint64_t>(rows) * blockIdx) / blockNum);
        uint32_t rowEnd   = static_cast<uint32_t>((static_cast<uint64_t>(rows) * (blockIdx + 1)) / blockNum);

        for (uint32_t r = rowStart; r < rowEnd; ++r) {
            // Pass 1: compute row max
            float rowMax = -FLT_MAX;
            for (uint32_t c = 0; c < cols; c += tileCols) {
                uint32_t len = AscendC::min(tileCols, cols - c);
                AscendC::LocalTensor<float> xLocal = bufX.Get<float>();
                AscendC::DataCopy(xLocal, xGm[r * cols + c], len);

                AscendC::LocalTensor<float> redLocal = bufRed1.Get<float>();
                AscendC::ReduceMax(redLocal, xLocal, len);
                // assume API to read first value from local tensor
                float tileMax = redLocal.GetValue(0);
                if (tileMax > rowMax) rowMax = tileMax;
            }

            // Pass 2: compute sum of exp(x - rowMax)
            float rowSum = 0.0f;
            for (uint32_t c = 0; c < cols; c += tileCols) {
                uint32_t len = AscendC::min(tileCols, cols - c);
                AscendC::LocalTensor<float> xLocal = bufX.Get<float>();
                AscendC::DataCopy(xLocal, xGm[r * cols + c], len);

                AscendC::LocalTensor<float> tmp = bufY.Get<float>();
                AscendC::Adds(tmp, xLocal, -rowMax, len);   // tmp = x - rowMax
                AscendC::Exp(tmp, tmp, len);                // tmp = exp(x - rowMax)

                AscendC::LocalTensor<float> redLocal = bufRed2.Get<float>();
                AscendC::ReduceSum(redLocal, tmp, len);
                float tileSum = redLocal.GetValue(0);
                rowSum += tileSum;
            }

            // logSumExp = log(rowSum) + rowMax
            float logSumExp = 0.0f;
            {
                AscendC::LocalTensor<float> redLocal = bufRed1.Get<float>();
                // write rowSum to first element and take ln via vector op
                redLocal.SetValue(0, rowSum);
                AscendC::Ln(redLocal, redLocal, 1);
                logSumExp = redLocal.GetValue(0) + rowMax;
            }

            // Pass 3: y = x - logSumExp
            for (uint32_t c = 0; c < cols; c += tileCols) {
                uint32_t len = AscendC::min(tileCols, cols - c);
                AscendC::LocalTensor<float> xLocal = bufX.Get<float>();
                AscendC::DataCopy(xLocal, xGm[r * cols + c], len);

                AscendC::LocalTensor<float> yLocal = bufY.Get<float>();
                AscendC::Adds(yLocal, xLocal, -logSumExp, len);

                AscendC::DataCopy(yGm[r * cols + c], yLocal, len);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufRed1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufRed2;

    AscendC::GlobalTensor<float> xGm, yGm;

    uint32_t rows;
    uint32_t cols;
    uint32_t tileCols;
    int32_t dim;
};

extern "C" __global__ __aicore__ void log_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLogSoftmax op;
    op.Init(x, y, tiling_data.rows, tiling_data.cols, tiling_data.tileCols, tiling_data.dim);
    op.Process();
}
