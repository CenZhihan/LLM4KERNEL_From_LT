#!/bin/bash
echo "[Ascend910B1] Generating MinGptNewGeluCustom_5305f2c97a461a7c1e758ef3238e9f0a ..."
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
res=$(opc $1 --main_func=min_gpt_new_gelu_custom --input_param=/workspace/LLM4KERNEL_From_LT/ascend_op_projects/MinGptNewGeluCustom/build_out/op_kernel/binary/ascend910b/gen/MinGptNewGeluCustom_5305f2c97a461a7c1e758ef3238e9f0a_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/MinGptNewGeluCustom_5305f2c97a461a7c1e758ef3238e9f0a.json ; then
  echo "$2/MinGptNewGeluCustom_5305f2c97a461a7c1e758ef3238e9f0a.json not generated!"
  exit 1
fi

if ! test -f $2/MinGptNewGeluCustom_5305f2c97a461a7c1e758ef3238e9f0a.o ; then
  echo "$2/MinGptNewGeluCustom_5305f2c97a461a7c1e758ef3238e9f0a.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating MinGptNewGeluCustom_5305f2c97a461a7c1e758ef3238e9f0a Done"
