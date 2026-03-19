mkb_adapter：把外部实验室的 165 个通过 kernel 接入你们现有评测环境（只生成文件，不执行评测）

## 背景（你们现有评测链路的关键契约）
你们的评测入口是仓库根目录的 evaluation_parallel.py，它会读取：
  output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}/{op}.txt
并把 txt 内容作为 response_txt 传给 eval_single_runner.py，再由 utils/evaluation_utils.eval_single 调用后端。

对于 language=ascendc，后端 backends/ascendc_backend.py 会调用 utils/ascend_compile_pipeline.ascend_compile。
ascend_compile 期望：response_txt（generated_code）在 exec 后，向 context 注入以下 6 个字符串变量：
  - project_json_src
  - host_tiling_src
  - host_operator_src
  - kernel_src
  - python_bind_src
  - model_src
并最终 exec(model_src) 让 context 中出现 ModelNew。

mkb_adapter 的作用就是把：
  ZWN_repo/EvoKernel_ops-kernelbench-910b/ops-kernelbench-910b/*Custom/
里面现成的源码（json / op_host / op_kernel / pybind op.cpp）转换成你们评测能直接消费的 {op}.txt。

## 你将得到什么
1) 一个 manifest：列出 compiled=true 且 correctness=true 的 kernel 目录（按 result.json 筛选），默认写到：
   ZWN_repo/EvoKernel_ops-kernelbench-910b/manifests/passing_165.json

2) 一批 {op}.txt 文件：每个文件是一段 generated_code（仅变量赋值），默认写到：
   output/ascendc/external165/0.0-1.0/external_lab/run0/{op}.txt
目录结构完全对齐 evaluation_parallel.py 的读取规则，因此不需要修改任何现有评测脚本。

## 用法（只生成文件；你要求我不直接执行，这里只说明）
在仓库根目录（evaluation_parallel.py 所在目录）执行：

1) 生成 manifest
  python3 -m ZWN_repo.EvoKernel_ops-kernelbench-910b.mkb_adapter.cli scan

2) 生成 .txt
  python3 -m ZWN_repo.EvoKernel_ops-kernelbench-910b.mkb_adapter.cli materialize \
    --strategy external165 --model_name external_lab --run_id 0 --temperature 0.0 --top_p 1.0

如需显式指定仓库根目录（output/ 所在位置），加：
  --mkb_repo_root /path/to/LLM4KERNEL_From_LT

## 验收清单（生成后不跑也能检查）
- manifests/passing_165.json 里 num_entries == 165
- 对每个 entry，sources 指向的文件都存在
- 生成的 {op}.txt 内容包含 6 个变量名：
  project_json_src/host_tiling_src/host_operator_src/kernel_src/python_bind_src/model_src

## 常见失败原因（跑评测时）
- 形状/类型强约束：外部 pybind 里有 TORCH_CHECK 固定 shape（例如 argmax 的 [128,4096,4095]）。
  若 reference 的 get_inputs 不满足，会在 correctness/运行时失败。
- attr 传参：部分 op 有 attr；mkb_adapter 会尝试读取 project_json 的 attr default_value 并注入 ModelNew。
  若缺默认值且 binding 要求必传，可能失败（需要单独为该 op 定制）。

