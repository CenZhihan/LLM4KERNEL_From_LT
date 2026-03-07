
#include "kernel_operator.h"
#define __NPU_TILING__
#include "leaky_relu_custom_tiling_data.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelLeakyRelu {
public:
    __aicore__ inline KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum, float negativeSlope)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->negativeSlope = static_cast<float>(negativeSlope);
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // alloc tensor from queue memory
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        // copy progress_th tile from global tensor to local tensor
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpTensor1 = tmpBuffer1.Get<float>();
        AscendC::LocalTensor<float> tmpTensor2 = tmpBuffer2.Get<float>();
        float inputVal = 0.0;
        AscendC::Maxs(tmpTensor1, xLocal, inputVal, this->tileLength);
        AscendC::Mins(tmpTensor2, xLocal, inputVal, this->tileLength);
        AscendC::Muls(tmpTensor2, tmpTensor2, this->negativeSlope, this->tileLength);
        AscendC::Add(yLocal, tmpTensor1, tmpTensor2, this->tileLength);
        // enque the output tensor to VECOUT queue
        outQueueY.EnQue<float>(yLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // deque output tensor from VECOUT queue
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        // copy progress_th tile from local tensor to global tensor
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        // free output tensor for reuse
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;
    // create queues for input, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float negativeSlope;
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum, tiling_data.negativeSlope);
    op.Process();
}
