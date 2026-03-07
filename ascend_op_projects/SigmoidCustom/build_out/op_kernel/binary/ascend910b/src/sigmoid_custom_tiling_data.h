#ifndef __SIGMOID_CUSTOM_TILING_H__
#define __SIGMOID_CUSTOM_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct SigmoidCustomTilingData {
    uint32_t totalLength = 0;
    uint32_t tileNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitSigmoidCustomTilingData(const __gm__ uint8_t* tiling, SigmoidCustomTilingData* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(SigmoidCustomTilingData) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitSigmoidCustomTilingData(uint8_t* tiling, SigmoidCustomTilingData* const_data)
{
    uint64_t *src = (uint64_t *)tiling;
    uint64_t *dst = (uint64_t *)const_data;
    for (auto i = 0; i < sizeof(SigmoidCustomTilingData) / 8; i++) *(dst + i) = *(src + i);
}
#endif


#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
SigmoidCustomTilingData tiling_data; \
InitSigmoidCustomTilingData(tiling_arg, &tiling_data)

#endif