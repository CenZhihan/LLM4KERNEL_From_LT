# Softmax 完整 Prompt 示例（拼完后给 LLM 的样子）

假设：`op=softmax`，策略 `add_shot`，`tool_mode=KB_ONLY`，因此会先拼「KB 说明」，再拼任务描述。  
下面整段 = 进入 Agent 时**第一条 HumanMessage 的 content**，也就是「用户问题」整段。

---

## 一、逐段拆解（按拼接顺序）

---

### 【Part 0】

**来源：** `Agent_kernel/agent_runner.py` 中写死的字符串（仅当 `tool_mode in (KB_ONLY, KB_AND_WEB)` 时加在最前）

```
【说明】请先使用知识库检索与本题相关的文档、API 说明或示例，再基于检索结果作答；不要仅凭已有知识直接写代码。

```

---

### 【Part 1】

**来源：** `prompt_generators/prompt_utils.py` 常量 `ASCENDC_PROBLEM_STATEMENT`

```
You are an expert in writing custom AscendC kernels to optimize PyTorch architectures by replacing specific operators for performance gains.

```

---

### 【Part 2】

**来源：** `prompt_utils.ascendc_template()` 里写死的英文说明 + 两个文件的**完整内容**

- 说明里的 kernel 名：`add_custom`（示例 op）
- 第一段代码块：`prompts/cuda_model_add.py` 全文
- 第二段代码块：`prompts/ascendc_new_model_add.py` 全文

```
    Here is an example to illustrate the expected transformation using custom AscendC operators. **Original architecture with kernel name `add_custom`:**

    ```python

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(2, 2048)
    b = torch.randn(2, 2048)
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []

    ```

    Transformed version using custom AscendC kernels:
    This transformation includes six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src` and `model_src`.
    The kernel function name in `kernel_src` must exactly match the provided kernel name. The operator definition in `project_json_src` and `host_operator_src` should also correspond to the kernel name, but follow PascalCase naming:

    ```python
project_json_src='''
[
    {
        "op": "AddCustom",
        ...
    }
]
'''
host_tiling_src="""
...
"""
host_operator_src="""
...
"""
kernel_src="""
...
"""
python_bind_src="""
...
"""
model_src="""
...
"""
    ```

```

（上面第二个 ```python 块里是 `prompts/ascendc_new_model_add.py` 的**完整文件内容**，包含六个变量的定义；这里用 `...` 省略中间部分，实际拼进去的是整份文件。）

---

### 【Part 3】

**来源：** `prompt_utils.ascendc_template()` 里写死的引导句 + **当前 op 的 reference 架构文件**

- 引导句里的名字：`softmax_custom`，PascalCase `SoftmaxCustom`（由 `op + '_custom'` 和 `underscore_to_pascalcase(op)` 得到）
- 代码块内容：`reference/{category}/{op}.py` = **`reference/activation/softmax.py` 全文**

```
    Now, you are given the following architecture with kernel name softmax_custom(PascalCase: SoftmaxCustom):

    ```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

# batch_size = 4096
# dim = 393216
batch_size = 512
dim = 32768

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
    ```

```

---

### 【Part 4】

**来源：** `prompt_generators/prompt_utils.py` 常量 `ASCENDC_PROBLEM_INSTRUCTION`

```
Your task: Replace relevant PyTorch operators in the architecture named Model with custom AscendC kernels. Generate an optimized version named ModelNew, including the six Python strings listed above. Just output the code, no other text, and NO testing code!

```

---

## 二、来源汇总表（softmax 用到的文件）

| 顺序 | 内容简述 | 来源（代码/文件） |
|------|----------|-------------------|
| 0 | KB 说明（先查知识库再写代码） | `agent_runner.py` 内写死 |
| 1 | 角色说明（AscendC 专家） | `prompt_utils.ASCENDC_PROBLEM_STATEMENT` |
| 2 | 示例：原始架构 | `prompts/cuda_model_add.py` 全文 |
| 2 | 示例：转换后（六段字符串） | `prompts/ascendc_new_model_add.py` 全文 |
| 3 | “Now you are given…” + 当前架构 | 引导句在 `prompt_utils.ascendc_template()`；代码块 = `reference/activation/softmax.py` 全文 |
| 4 | 任务指令（生成 ModelNew 与六字符串） | `prompt_utils.ASCENDC_PROBLEM_INSTRUCTION` |

---

## 三、整段连起来长什么样（无省略版结构）

下面把「Part 0～4」按真实拼接顺序连成一段，长文件用「<文件路径>」标出，便于对照：

```
【说明】请先使用知识库检索与本题相关的文档、API 说明或示例，再基于检索结果作答；不要仅凭已有知识直接写代码。

You are an expert in writing custom AscendC kernels to optimize PyTorch architectures by replacing specific operators for performance gains.

    Here is an example to illustrate the expected transformation using custom AscendC operators. **Original architecture with kernel name `add_custom`:**

    ```python
<prompts/cuda_model_add.py 全文>
    ```

    Transformed version using custom AscendC kernels:
    This transformation includes six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src` and `model_src`.
    The kernel function name in `kernel_src` must exactly match the provided kernel name. The operator definition in `project_json_src` and `host_operator_src` should also correspond to the kernel name, but follow PascalCase naming:

    ```python
<prompts/ascendc_new_model_add.py 全文>
    ```

    Now, you are given the following architecture with kernel name softmax_custom(PascalCase: SoftmaxCustom):

    ```python
<reference/activation/softmax.py 全文>
    ```

Your task: Replace relevant PyTorch operators in the architecture named Model with custom AscendC kernels. Generate an optimized version named ModelNew, including the six Python strings listed above. Just output the code, no other text, and NO testing code!
```

上面这一段，就是「拼完之后、进入 LLM」的完整 prompt 的样子；每部分的文字来源就是前面 Part 0～4 和表格里列出的那些。若要看某一段的逐字内容，直接打开对应文件即可（尤其是 `cuda_model_add.py`、`ascendc_new_model_add.py`、`reference/activation/softmax.py`）。
