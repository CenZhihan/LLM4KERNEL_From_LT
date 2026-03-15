import os
from config import project_root_path
from dataset import dataset
from utils.utils import read_file, underscore_to_pascalcase

template_statement="""You write custom {} kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom {} kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
template_instruction="""
Optimize the architecture named Model with custom {} operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

template_example_intro='''
Here's an example to show you the syntax of inline embedding custom {} operators in torch: The example given architecture is:
'''

template_new_arch_intro='''
The example new arch with custom {} kernels looks like this:
'''

ASCENDC_PROBLEM_STATEMENT = 'You are an expert in writing custom AscendC kernels to optimize PyTorch architectures by replacing specific operators for performance gains.\n'
ASCENDC_PROBLEM_INSTRUCTION='''
Your task: Replace relevant PyTorch operators in the architecture named Model with custom AscendC kernels. Generate an optimized version named ModelNew, including the six Python strings listed above. Just output the code, no other text, and NO testing code!\n
'''

# none 策略下无示例时，必须显式说明期望的输出格式，否则模型不知道要生成六个变量
# 与 add_shot/selected_shot 的区别：不提供任何示例代码，仅通过本段文字规范输出格式与结构
ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE = """
Output format (strict, no examples given here—follow these rules only):

1) You must output exactly one code block: start with a line ```python and end with ```. Do NOT output any prose, description, ellipsis (e.g. "、、、"), or markdown before or after this block. Do NOT put any non-Python line inside the block.

2) The block must be valid, executable Python: use 4 spaces for indentation (no tabs); no syntax errors; no trailing comment lines that look like prose. Use only ASCII in the entire block (no Chinese characters or fullwidth punctuation such as 。、（，）；write all comments and docstrings in English). The entire block will be run as Python to extract the six variables.

3) The block must define exactly these six string variables (e.g. using r\"\"\" or r''' for multi-line strings):
- project_json_src: JSON string for the operator project. The parsed JSON must have a top-level key \"op\" (required by the Ascend build toolchain). Under \"op\" put operator metadata (name, inputs, outputs, kernel/tiling/sources, etc.). Do not use other top-level shapes like \"op_name\" or \"operator\" alone—the root must contain \"op\".
- host_tiling_src: Tiling header (.h) content
- host_operator_src: Host-side operator (.cpp) content
- kernel_src: AscendC kernel (.cpp) content (kernel name must match the given operator name)
- python_bind_src: Python binding code
- model_src: PyTorch model code that uses the custom op (class ModelNew)

4) Naming: kernel and operator names must use the given name (e.g. {op_name}) and PascalCase in JSON/host (e.g. {op_pascal}).
"""

def read_relavant_files(language, op, example):
    category = dataset[op]['category']
    example_arch_path = os.path.join(
        project_root_path, f"prompts/cuda_model_{example}.py"
    )
    example_new_arch_path = os.path.join(
        project_root_path, f"prompts/{language}_new_model_{example}.py"
    )
    new_arch_path = os.path.join(
        project_root_path, f"reference/{category}/{op}.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )
    if not os.path.exists(new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {new_arch_path}"
        )
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    arch = read_file(new_arch_path)
    return arch, example_arch, example_new_arch

def generate_template(arc_src, example_arch_src, example_new_arch_src, language):
    prompt = template_statement.format(language, language)

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        {template_example_intro.format(language)} \n
        ``` \n
        {example_arch_src}
        ``` \n
        {template_new_arch_intro.format(language)} 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += template_instruction.format(language)
    return prompt    

def ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, example_op):
        # add custom to name to prevent conficts with existing operators
        op = op + '_custom'
        example_op = example_op + '_custom'
        prompt = ASCENDC_PROBLEM_STATEMENT

        if example_arch_src != "" and example_new_arch_src != "":
            prompt += f"""
    Here is an example to illustrate the expected transformation using custom AscendC operators. **Original architecture with kernel name `{example_op}`:**\n
    ```python \n
    {example_arch_src}
    ``` \n
    Transformed version using custom AscendC kernels:
    This transformation includes six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src` and `model_src`.
    The kernel function name in `kernel_src` must exactly match the provided kernel name. The operator definition in `project_json_src` and `host_operator_src` should also correspond to the kernel name, but follow PascalCase naming: 
    ```python
    {example_new_arch_src}
    ``` \n
    """
        else:
            # none 策略：没有示例时显式列出六个变量的要求，避免模型不知道输出格式
            prompt += ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE.format(
                op_name=op,
                op_pascal=underscore_to_pascalcase(op),
            )

        prompt += f"""
    Now, you are given the following architecture with kernel name {op}(PascalCase: {underscore_to_pascalcase(op)}): \n
    ```python
    {arc_src}
    ```
        """
        prompt += ASCENDC_PROBLEM_INSTRUCTION
        return prompt