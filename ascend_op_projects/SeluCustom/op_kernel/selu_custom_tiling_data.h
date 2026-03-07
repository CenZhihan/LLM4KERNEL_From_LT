#ifndef __SELU_CUSTOM_TILING_H__
#define __SELU_CUSTOM_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct SeluCustomTilingData {
    uint32_t totalLength = 0;
    uint32_t tileNum = 0;
    float alpha = 0;
    float scale = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitSeluCustomTilingData(const __gm__ uint8_t* tiling, SeluCustomTilingData* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(SeluCustomTilingData) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitSeluCustomTilingData(uint8_t* tiling, SeluCustomTilingData* const_data)
{
    uint64_t *src = (uint64_t *)tiling;
    uint64_t *dst = (uint64_t *)const_data;
    for (auto i = 0; i < sizeof(SeluCustomTilingData) / 8; i++) *(dst + i) = *(src + i);
}
#endif


#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
SeluCustomTilingData tiling_data; \
InitSeluCustomTilingData(tiling_arg, &tiling_data)

#endif