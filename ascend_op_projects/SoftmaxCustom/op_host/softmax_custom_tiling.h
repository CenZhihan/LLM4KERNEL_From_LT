
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalRows);
  TILING_DATA_FIELD_DEF(uint32_t, rowSize);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxCustomTilingData)
}
