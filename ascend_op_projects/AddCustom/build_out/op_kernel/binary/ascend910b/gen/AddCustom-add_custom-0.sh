#!/bin/bash
echo "[Ascend910B1] Generating AddCustom_6b4edd934cf3a660c0cd2e16b3659524 ..."
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
res=$(opc $1 --main_func=add_custom --input_param=/workspace/LLM4KERNEL_From_LT/ascend_op_projects/AddCustom/build_out/op_kernel/binary/ascend910b/gen/AddCustom_6b4edd934cf3a660c0cd2e16b3659524_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/AddCustom_6b4edd934cf3a660c0cd2e16b3659524.json ; then
  echo "$2/AddCustom_6b4edd934cf3a660c0cd2e16b3659524.json not generated!"
  exit 1
fi

if ! test -f $2/AddCustom_6b4edd934cf3a660c0cd2e16b3659524.o ; then
  echo "$2/AddCustom_6b4edd934cf3a660c0cd2e16b3659524.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating AddCustom_6b4edd934cf3a660c0cd2e16b3659524 Done"
