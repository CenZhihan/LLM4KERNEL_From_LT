// Standalone test: dump the host-side tiling buffer layout to verify
// if there is an 8-byte placeholder/alignment gap before the actual fields.
// Compile inside container:
//   g++ -std=c++17 -I/usr/local/Ascend/ascend-toolkit/latest/include \
//       -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
//       -lascendcl -lregister -o diag_tiling_layout diag_tiling_layout.cpp
// Or just compile with the same include paths as the build system.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, fullTilesPerBlock);
  TILING_DATA_FIELD_DEF(uint32_t, hasTail);
END_TILING_DATA_DEF;
}

int main() {
    optiling::ReluCustomTilingData tiling;

    tiling.set_totalLength(0xAAAAAAAA);
    tiling.set_tileLength(0xBBBBBBBB);
    tiling.set_blockDim(0xCCCCCCCC);
    tiling.set_fullTilesPerBlock(0xDDDDDDDD);
    tiling.set_hasTail(0xEEEEEEEE);

    size_t bufSize = tiling.GetDataSize();
    printf("GetDataSize() = %zu bytes\n", bufSize);

    uint8_t buf[256] = {};
    tiling.SaveToBuffer(buf, sizeof(buf));

    printf("Buffer hex dump (%zu bytes):\n", bufSize);
    for (size_t i = 0; i < bufSize; i++) {
        printf("%02X ", buf[i]);
        if ((i + 1) % 4 == 0) printf("  ");
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n\nLooking for markers:\n");
    for (size_t i = 0; i + 3 < bufSize; i += 4) {
        uint32_t val;
        memcpy(&val, buf + i, 4);
        const char* label = "";
        if (val == 0xAAAAAAAA) label = " <-- totalLength";
        else if (val == 0xBBBBBBBB) label = " <-- tileLength";
        else if (val == 0xCCCCCCCC) label = " <-- blockDim";
        else if (val == 0xDDDDDDDD) label = " <-- fullTilesPerBlock";
        else if (val == 0xEEEEEEEE) label = " <-- hasTail";
        if (*label) {
            printf("  offset %3zu: 0x%08X%s\n", i, val, label);
        }
    }

    printf("\n--- Device-side struct would read ---\n");
    // Device side: #pragma pack(1) struct with 5 uint32_t = 20 bytes
    // Init copies sizeof(struct)/4 = 5 uint32_t values from buffer start
    for (int i = 0; i < 5; i++) {
        uint32_t val;
        memcpy(&val, buf + i * 4, 4);
        const char* names[] = {"totalLength", "tileLength", "blockDim", "fullTilesPerBlock", "hasTail"};
        printf("  device %s (offset %d) = 0x%08X\n", names[i], i * 4, val);
    }

    return 0;
}
