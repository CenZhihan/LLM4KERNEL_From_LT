from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from prompt_generators.prompt_utils import ascendc_template
from config import project_root_path
from dataset import dataset
from utils.utils import read_file
import os


@register_prompt("ascendc", "none")
class AscendcNoShotPromptStrategy(BasePromptStrategy):
    """
    AscendC 策略：不提供示例，只给当前算子的参考实现。
    """

    def generate(self, op: str) -> str:
        category = dataset[op]["category"]
        arch_path = os.path.join(
            project_root_path,
            f"reference/{category}/{op}.py",
        )
        if not os.path.exists(arch_path):
            raise FileNotFoundError(f"Reference architecture file not found: {arch_path}")
        arc_src = read_file(arch_path)
        # 不给示例，直接传空字符串，ascendc_template 内部会自动跳过 example 段落
        example_arch_src = ""
        example_new_arch_src = ""
        return ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, op)

