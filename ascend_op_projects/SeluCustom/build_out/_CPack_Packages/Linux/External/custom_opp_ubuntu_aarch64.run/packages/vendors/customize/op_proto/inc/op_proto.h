#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SeluCustom)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(alpha, Float, 1.67326)
    .ATTR(scale, Float, 1.0507)
    .OP_END_FACTORY_REG(SeluCustom);

}

#endif
