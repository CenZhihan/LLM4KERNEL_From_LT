
#include "kernel_operator.h"
#define __NPU_TILING__
#include "min_gpt_new_gelu_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr float COEF_C = 0.044715f;
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f; // sqrt(2/pi)

class KernelMinGptNewGelu {
public:
    __aicore__ inline KernelMinGptNewGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer4, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t /*progress*/)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        AscendC::LocalTensor<float> tmp1 = tmpBuffer1.Get<float>(); // x^2
        AscendC::LocalTensor<float> tmp2 = tmpBuffer2.Get<float>(); // x^3 scaled
        AscendC::LocalTensor<float> tmp3 = tmpBuffer3.Get<float>(); // inner and scaled
        AscendC::LocalTensor<float> tmp4 = tmpBuffer4.Get<float>(); // tanh and 1+tanh

        // tmp1 = x * x
        AscendC::Mul(tmp1, xLocal, xLocal, this->tileLength);
        // tmp2 = tmp1 * x = x^3
        AscendC::Mul(tmp2, tmp1, xLocal, this->tileLength);
        // tmp2 = 0.044715 * x^3
        AscendC::Muls(tmp2, tmp2, COEF_C, this->tileLength);
        // tmp3 = x + 0.044715*x^3
        AscendC::Add(tmp3, xLocal, tmp2, this->tileLength);
        // tmp3 = sqrt(2/pi) * tmp3
        AscendC::Muls(tmp3, tmp3, SQRT_2_OVER_PI, this->tileLength);
        // tmp4 = tanh(tmp3)
        AscendC::Tanh(tmp4, tmp3, this->tileLength);
        // tmp4 = 1 + tanh(...)
        AscendC::Adds(tmp4, tmp4, 1.0f, this->tileLength);
        // tmp3 = 0.5 * tmp4
        AscendC::Muls(tmp3, tmp4, 0.5f, this->tileLength);
        // y = x * tmp3
        AscendC::Mul(yLocal, xLocal, tmp3, this->tileLength);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3, tmpBuffer4;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void min_gpt_new_gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMinGptNewGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
