
#include "kernel_operator.h"

#define __NPU_TILING__
#include "hardsigmoid_custom_tiling_data.h"
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelHardsigmoid {
public:
    __aicore__ inline KernelHardsigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // HardSigmoid: y = min(max((x + 3) / 6, 0), 1)
        AscendC::LocalTensor<DTYPE_Y> tmp1 = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> tmp2 = outQueueY.AllocTensor<DTYPE_Y>();

        // tmp1 = x + 3
        AscendC::AddScalar(tmp1, xLocal, (DTYPE_X)3.0, this->tileLength);
        // tmp2 = tmp1 / 6
        AscendC::DivScalar(tmp2, tmp1, (DTYPE_X)6.0, this->tileLength);
        // tmp1 = max(tmp2, 0)
        AscendC::MaximumScalar(tmp1, tmp2, (DTYPE_X)0.0, this->tileLength);
        // yLocal = min(tmp1, 1)
        AscendC::MinimumScalar(yLocal, tmp1, (DTYPE_X)1.0, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);

        inQueueX.FreeTensor(xLocal);
        outQueueY.FreeTensor(tmp1);
        outQueueY.FreeTensor(tmp2);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void hardsigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelHardsigmoid op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
