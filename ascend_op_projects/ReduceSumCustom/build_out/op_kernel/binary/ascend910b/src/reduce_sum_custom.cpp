
#include "kernel_operator.h"
#define __NPU_TILING__
#include "reduce_sum_custom_tiling_data.h"

#define REDUCE_TILING_0 1
#define REDUCE_TILING_1 2
#define REDUCE_TILING_2 3

class KernelReduce {
static constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
static constexpr uint32_t DEFAULT_REP_STRIDE = 8;
static constexpr uint32_t REP_LEN = 256;
static constexpr uint32_t BLK_LEN = 32;
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t outLength)
    {
        this->totalLength = totalLength;
        this->outLength = outLength;

        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        zGm.SetGlobalBuffer((__gm__ float *)z, outLength);
        pipe.InitBuffer(inQueueX, 1, totalLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, outLength * sizeof(float));
    }
    __aicore__ inline void Process1()
    {
        CopyIn();
        Compute1();
        CopyOut();
    }
    __aicore__ inline void Process2()
    {
        CopyIn();
        Compute2();
        CopyOut();
    }
    __aicore__ inline void Process3()
    {
        CopyIn();
        Compute3();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm, totalLength);
        inQueueX.EnQue(xLocal);
    }
    // Only WholeReduceSum is used under 256B.
    __aicore__ inline void Compute1()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        constexpr int64_t maskLen = REP_LEN / sizeof(float);
        AscendC::WholeReduceSum<float>(zLocal, xLocal, maskLen, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    // One WholeReduceSum and one BlockReduceSum are used in (256B,2KB](for float input).
    __aicore__ inline void Compute2()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        pipe.InitBuffer(calcBuf, totalLength * sizeof(float));
        AscendC::LocalTensor<float> tempTensor1 = calcBuf.Get<float>();
        constexpr uint32_t c0Count = BLK_LEN / sizeof(float);
        const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float>(0, totalLength);
        AscendC::BlockReduceSum<float, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(0, blockNum0);
        AscendC::WholeReduceSum<float, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    // Two WholeReduceSum are used in (2KB,16KB](for float input).
    __aicore__ inline void Compute3()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        pipe.InitBuffer(calcBuf, totalLength * sizeof(float));
        AscendC::LocalTensor<float> tempTensor1 = calcBuf.Get<float>();
        const uint32_t repeatNum = (totalLength * sizeof(float) + REP_LEN - 1) / REP_LEN;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float>(0, totalLength);
        AscendC::WholeReduceSum<float, false>(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(0, repeatNum);
        AscendC::WholeReduceSum<float, false>(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm, zLocal, this->outLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t totalLength;
    uint32_t outLength;
};

extern "C" __global__ __aicore__ void reduce_sum_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelReduce op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.outLength);
    if (TILING_KEY_IS(REDUCE_TILING_0)) {
        op.Process1();
    } else if (TILING_KEY_IS(REDUCE_TILING_1)) {
        op.Process2();
    } else if (TILING_KEY_IS(REDUCE_TILING_2)) {
        op.Process3();
    }
}
