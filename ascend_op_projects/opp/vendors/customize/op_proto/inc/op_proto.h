#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(Conv2dAddScaleSigmoidGroupNormCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .INPUT(scale, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .INPUT(beta, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(Conv2dAddScaleSigmoidGroupNormCustom);

}

#endif
