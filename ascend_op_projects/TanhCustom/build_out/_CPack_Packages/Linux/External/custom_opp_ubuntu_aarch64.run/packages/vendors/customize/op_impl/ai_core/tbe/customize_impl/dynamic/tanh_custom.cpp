
#include "kernel_operator.h"
#define __NPU_TILING__
#include "tanh_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(float)); // exp(+x) or exp(2x) intermediate
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float)); // exp(-x) intermediate
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(float)); // numerator
        pipe.InitBuffer(tmpBuffer4, this->tileLength * sizeof(float)); // denominator
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

        AscendC::LocalTensor<float> tmpExpPos = tmpBuffer1.Get<float>();
        AscendC::LocalTensor<float> tmpExpNeg = tmpBuffer2.Get<float>();
        AscendC::LocalTensor<float> tmpNum = tmpBuffer3.Get<float>();
        AscendC::LocalTensor<float> tmpDen = tmpBuffer4.Get<float>();

        // Compute tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
        AscendC::Exp(tmpExpPos, xLocal, this->tileLength);                      // e^x
        AscendC::Muls(tmpExpNeg, xLocal, -1.0f, this->tileLength);             // -x
        AscendC::Exp(tmpExpNeg, tmpExpNeg, this->tileLength);                  // e^{-x}
        AscendC::Sub(tmpNum, tmpExpPos, tmpExpNeg, this->tileLength);          // numerator
        AscendC::Add(tmpDen, tmpExpPos, tmpExpNeg, this->tileLength);          // denominator
        AscendC::Div(yLocal, tmpNum, tmpDen, this->tileLength);                // y = num / den

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

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelTanh op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
