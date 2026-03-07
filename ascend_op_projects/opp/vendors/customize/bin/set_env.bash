#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/workspace/LLM4KERNEL_From_LT/ascend_op_projects/opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/workspace/LLM4KERNEL_From_LT/ascend_op_projects/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
