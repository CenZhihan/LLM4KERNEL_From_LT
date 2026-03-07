#!/bin/bash
echo "[Ascend910B1] Generating SoftplusCustom_4f98c38c51fc1ba3d8de7343db4dead2 ..."
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
res=$(opc $1 --main_func=softplus_custom --input_param=/workspace/LLM4KERNEL_From_LT/ascend_op_projects/SoftplusCustom/build_out/op_kernel/binary/ascend910b/gen/SoftplusCustom_4f98c38c51fc1ba3d8de7343db4dead2_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/SoftplusCustom_4f98c38c51fc1ba3d8de7343db4dead2.json ; then
  echo "$2/SoftplusCustom_4f98c38c51fc1ba3d8de7343db4dead2.json not generated!"
  exit 1
fi

if ! test -f $2/SoftplusCustom_4f98c38c51fc1ba3d8de7343db4dead2.o ; then
  echo "$2/SoftplusCustom_4f98c38c51fc1ba3d8de7343db4dead2.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating SoftplusCustom_4f98c38c51fc1ba3d8de7343db4dead2 Done"
