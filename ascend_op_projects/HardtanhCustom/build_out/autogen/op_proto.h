#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(HardtanhCustom)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(min_val, Float, -1)
    .ATTR(max_val, Float, 1)
    .OP_END_FACTORY_REG(HardtanhCustom);

}

#endif
