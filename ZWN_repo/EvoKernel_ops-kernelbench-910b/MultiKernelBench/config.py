import os

# path
# 获取 MultiKernelBench 所在目录
_multikernelbench_dir = os.path.dirname(os.path.abspath(__file__))
# 根目录 (LLM4KERNEL_From_LT)
project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(_multikernelbench_dir)))
ref_impl_base_path = f'{_multikernelbench_dir}/reference'
# Virtual include path placeholder for Catlass backend.
catlass_include_path = "/virtual/path/to/catlass/include"

# trial
max_turn = 1
num_correct_trials = 5
num_perf_trials = 100
num_warmup = 3

# LLM config
max_tokens = 8192
temperature = 0.0
top_p=1.0
num_completions=1

seed_num=1024

# cuda device
arch_list = ['Ada']
arch_list_xpu = ['dg2']

# Ascend compile related
op_engineer_dir = f'{project_root_path}/ascend_op_projects'
deploy_path = f'{op_engineer_dir}/opp'
ascendc_device = 'ai_core-Ascend910B2'
