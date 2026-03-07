#!/bin/bash
echo "[Ascend910B1] Generating SoftmaxCustom_2b1468cd6bb3e5bee35b792533f987bc ..."
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
res=$(opc $1 --main_func=softmax_custom --input_param=/workspace/LLM4KERNEL_From_LT/ascend_op_projects/SoftmaxCustom/build_out/op_kernel/binary/ascend910b/gen/SoftmaxCustom_2b1468cd6bb3e5bee35b792533f987bc_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/SoftmaxCustom_2b1468cd6bb3e5bee35b792533f987bc.json ; then
  echo "$2/SoftmaxCustom_2b1468cd6bb3e5bee35b792533f987bc.json not generated!"
  exit 1
fi

if ! test -f $2/SoftmaxCustom_2b1468cd6bb3e5bee35b792533f987bc.o ; then
  echo "$2/SoftmaxCustom_2b1468cd6bb3e5bee35b792533f987bc.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating SoftmaxCustom_2b1468cd6bb3e5bee35b792533f987bc Done"
