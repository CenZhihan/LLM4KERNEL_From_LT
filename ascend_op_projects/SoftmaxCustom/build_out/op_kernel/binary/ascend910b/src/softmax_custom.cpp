
#include "kernel_operator.h"
#define __NPU_TILING__
#include "softmax_custom_tiling_data.h"
#include <cmath>

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

typedef float DTYPE_X;
typedef float DTYPE_Z;

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum, uint32_t feature)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->feature = feature;
        // tileLength is number of elements per tile
        this->tileLength = this->blockLength / tileNum;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
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

        // number of rows in this tile (each row has 'feature' elements)
        uint32_t rows = 1;
        if (this->feature > 0) {
            rows = this->tileLength / this->feature;
        }

        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t base = r * this->feature;
            // compute max for numerical stability
            DTYPE_X maxv = xLocal[base];
            for (uint32_t j = 1; j < this->feature; ++j) {
                DTYPE_X val = xLocal[base + j];
                if (val > maxv) maxv = val;
            }
            // exponentiate and accumulate sum
            DTYPE_Z sum = 0.0f;
            for (uint32_t j = 0; j < this->feature; ++j) {
                DTYPE_Z v = expf((DTYPE_Z)(xLocal[base + j] - maxv));
                zLocal[base + j] = v;
                sum += v;
            }
            // normalize
            if (sum == 0.0f) sum = 1.0f;
            for (uint32_t j = 0; j < this->feature; ++j) {
                zLocal[base + j] = zLocal[base + j] / sum;
            }
        }

        outQueueZ.EnQue(zLocal);
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
    uint32_t feature;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.feature);
    op.Process();
}
