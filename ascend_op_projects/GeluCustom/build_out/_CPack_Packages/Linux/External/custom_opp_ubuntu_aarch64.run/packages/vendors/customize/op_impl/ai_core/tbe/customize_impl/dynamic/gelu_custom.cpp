
#include "kernel_operator.h"
#define __NPU_TILING__
#include "gelu_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Z *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
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

    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> yLocal = outQueueY.AllocTensor<DTYPE_Z>();

        // GELU approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        // Step1: yLocal = x^2
        AscendC::Mul(yLocal, xLocal, xLocal, this->tileLength);
        // Step2: yLocal = x^3
        AscendC::Mul(yLocal, yLocal, xLocal, this->tileLength);
        // Step3: yLocal = 0.044715 * x^3
        AscendC::Muls(yLocal, yLocal, static_cast<DTYPE_Z>(0.044715f), this->tileLength);
        // Step4: yLocal = x + 0.044715 * x^3
        AscendC::Add(yLocal, yLocal, xLocal, this->tileLength);
        // Step5: yLocal = sqrt(2/pi) * (x + 0.044715 * x^3)
        AscendC::Muls(yLocal, yLocal, static_cast<DTYPE_Z>(0.7978845608028654f), this->tileLength); // sqrt(2/pi)
        // Step6: yLocal = tanh(yLocal)
        AscendC::Tanh(yLocal, yLocal, this->tileLength);
        // Step7: yLocal = 1 + yLocal
        AscendC::Adds(yLocal, yLocal, static_cast<DTYPE_Z>(1.0f), this->tileLength);
        // Step8: yLocal = 0.5 * yLocal
        AscendC::Muls(yLocal, yLocal, static_cast<DTYPE_Z>(0.5f), this->tileLength);
        // Step9: yLocal = x * yLocal
        AscendC::Mul(yLocal, yLocal, xLocal, this->tileLength);

        outQueueY.EnQue<DTYPE_Z>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> yLocal = outQueueY.DeQue<DTYPE_Z>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Z> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
