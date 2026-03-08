
#include "kernel_operator.h"
#define __NPU_TILING__
#include "min_gpt_new_gelu_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr float K_C0 = 0.044715f;
constexpr float K_S = 0.7978845608028654f; // sqrt(2/pi)
constexpr float K_HALF = 0.5f;

class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
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

    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // yLocal will be used as temp and final output buffer.
        // Step1: y = x*x
        AscendC::Mul(yLocal, xLocal, xLocal, this->tileLength);
        // Step2: y = y*x = x^3
        AscendC::Mul(yLocal, yLocal, xLocal, this->tileLength);
        // Step3: y = 0.044715 * y
        AscendC::Muls(yLocal, yLocal, K_C0, this->tileLength);
        // Step4: y = y + x
        AscendC::Add(yLocal, yLocal, xLocal, this->tileLength);
        // Step5: y = sqrt(2/pi) * y
        AscendC::Muls(yLocal, yLocal, K_S, this->tileLength);
        // Step6: y = tanh(y)
        AscendC::Tanh(yLocal, yLocal, this->tileLength);
        // Step7: y = y + 1
        AscendC::Adds(yLocal, yLocal, 1.0f, this->tileLength);
        // Step8: y = y * x
        AscendC::Mul(yLocal, yLocal, xLocal, this->tileLength);
        // Step9: y = 0.5 * y
        AscendC::Muls(yLocal, yLocal, K_HALF, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
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

extern "C" __global__ __aicore__ void min_gpt_new_gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
