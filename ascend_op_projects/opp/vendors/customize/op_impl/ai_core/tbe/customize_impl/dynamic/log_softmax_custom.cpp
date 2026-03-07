
#include "kernel_operator.h"
#define __NPU_TILING__
#include "log_softmax_custom_tiling_data.h"
#include <cmath>

constexpr int32_t BUFFER_NUM = 1; // single buffer

class KernelLogSoftmax {
public:
    __aicore__ inline KernelLogSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t dim, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        // set GM buffers aligned to this block
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));

        this->dim = dim;
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
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

        uint32_t numElements = this->tileLength;
        uint32_t d = this->dim;
        // guard division
        if (d == 0) d = 1;
        uint32_t rows = numElements / d;

        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t rowOffset = r * d;
            // compute max
            float maxv = xLocal[rowOffset];
            for (uint32_t j = 1; j < d; ++j) {
                float v = xLocal[rowOffset + j];
                if (v > maxv) maxv = v;
            }
            // compute sum of exp(x - max)
            double sumexp = 0.0;
            for (uint32_t j = 0; j < d; ++j) {
                double ex = std::exp((double)xLocal[rowOffset + j] - (double)maxv);
                sumexp += ex;
                // store temporarily in zLocal to avoid extra buffer (will finalize next pass)
                zLocal[rowOffset + j] = (DTYPE_Z)ex;
            }
            double logsum = std::log(sumexp);
            // finalize outputs: x - max - logsum
            for (uint32_t j = 0; j < d; ++j) {
                float outv = (float)((double)std::log((double)zLocal[rowOffset + j]) + (double)maxv - logsum); 
                // equivalently: x - max - logsum. Use original xLocal for exactness.
                outv = xLocal[rowOffset + j] - maxv - (float)logsum;
                zLocal[rowOffset + j] = (DTYPE_Z)outv;
            }
        }

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t dim;
};

extern "C" __global__ __aicore__ void log_softmax_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLogSoftmax op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.dim, tiling_data.tileNum);
    op.Process();
}
