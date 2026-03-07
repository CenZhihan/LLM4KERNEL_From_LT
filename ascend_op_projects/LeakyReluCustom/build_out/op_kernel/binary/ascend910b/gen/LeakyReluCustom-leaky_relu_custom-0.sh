#!/bin/bash
echo "[Ascend910B1] Generating LeakyReluCustom_8cfa33d3cb332de6b8b432a5780d77ba ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=leaky_relu_custom --input_param=/workspace/LLM4KERNEL_From_LT/ascend_op_projects/LeakyReluCustom/build_out/op_kernel/binary/ascend910b/gen/LeakyReluCustom_8cfa33d3cb332de6b8b432a5780d77ba_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/LeakyReluCustom_8cfa33d3cb332de6b8b432a5780d77ba.json ; then
  echo "$2/LeakyReluCustom_8cfa33d3cb332de6b8b432a5780d77ba.json not generated!"
  exit 1
fi

if ! test -f $2/LeakyReluCustom_8cfa33d3cb332de6b8b432a5780d77ba.o ; then
  echo "$2/LeakyReluCustom_8cfa33d3cb332de6b8b432a5780d77ba.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating LeakyReluCustom_8cfa33d3cb332de6b8b432a5780d77ba Done"
