#ifndef __GELU_CUSTOM_TILING_H__
#define __GELU_CUSTOM_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct GeluCustomTilingData {
    uint32_t totalLength = 0;
    uint32_t tileNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline [aicore] void InitGeluCustomTilingData(const __gm__ uint8_t* tiling, GeluCustomTilingData* const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(GeluCustomTilingData) / 4; i++) *(dst + i) = *(src + i);
}
#else
inline void InitGeluCustomTilingData(uint8_t* tiling, GeluCustomTilingData* const_data)
{
    uint64_t *src = (uint64_t *)tiling;
    uint64_t *dst = (uint64_t *)const_data;
    for (auto i = 0; i < sizeof(GeluCustomTilingData) / 8; i++) *(dst + i) = *(src + i);
}
#endif


#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
GeluCustomTilingData tiling_data; \
InitGeluCustomTilingData(tiling_arg, &tiling_data)

#endif