#ifndef __SOFTMAX_CUSTOM_TILING_H__
#define __SOFTMAX_CUSTOM_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct SoftmaxCustomTilingData {
    uint32_t totalLength = 0;
    uint32_t tileNum = 0;
    uint32_t feature = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitSoftmaxCustomTilingData(const __gm__ uint8_t* tiling, SoftmaxCustomTilingData* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(SoftmaxCustomTilingData) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitSoftmaxCustomTilingData(uint8_t* tiling, SoftmaxCustomTilingData* const_data)
{
    uint64_t *src = (uint64_t *)tiling;
    uint64_t *dst = (uint64_t *)const_data;
    for (auto i = 0; i < sizeof(SoftmaxCustomTilingData) / 8; i++) *(dst + i) = *(src + i);
}
#endif


#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
SoftmaxCustomTilingData tiling_data; \
InitSoftmaxCustomTilingData(tiling_arg, &tiling_data)

#endif