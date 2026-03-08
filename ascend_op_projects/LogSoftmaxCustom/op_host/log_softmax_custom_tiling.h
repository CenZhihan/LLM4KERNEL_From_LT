
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogSoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rows);
  TILING_DATA_FIELD_DEF(uint32_t, cols);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(int32_t, axis);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogSoftmaxCustom, LogSoftmaxCustomTilingData)
}
