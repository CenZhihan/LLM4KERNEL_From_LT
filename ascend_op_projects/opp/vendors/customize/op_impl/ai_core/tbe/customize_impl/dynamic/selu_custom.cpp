
#include "kernel_operator.h"
#define __NPU_TILING__
#include "selu_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelSelu {
public:
    __aicore__ inline KernelSelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum, float alpha, float scale)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->alpha = static_cast<float>(alpha);
        this->scale = static_cast<float>(scale);
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> tmpPos = tmpBuffer1.Get<float>();
        AscendC::LocalTensor<float> tmpNeg = tmpBuffer2.Get<float>();

        float zero = 0.0f;

        // tmpPos = max(x, 0)
        AscendC::Maxs(tmpPos, xLocal, zero, this->tileLength);
        // tmpNeg = min(x, 0)
        AscendC::Mins(tmpNeg, xLocal, zero, this->tileLength);
        // tmpNeg = exp(tmpNeg) - 1
        AscendC::Exp(tmpNeg, tmpNeg, this->tileLength);
        AscendC::Adds(tmpNeg, tmpNeg, -1.0f, this->tileLength);
        // tmpNeg = alpha * tmpNeg
        AscendC::Muls(tmpNeg, tmpNeg, this->alpha, this->tileLength);
        // tmpNeg = tmpPos + tmpNeg
        AscendC::Add(tmpNeg, tmpPos, tmpNeg, this->tileLength);
        // y = scale * tmpNeg
        AscendC::Muls(yLocal, tmpNeg, this->scale, this->tileLength);

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
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float alpha;
    float scale;
};

extern "C" __global__ __aicore__ void selu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum, tiling_data.alpha, tiling_data.scale);
    op.Process();
}
