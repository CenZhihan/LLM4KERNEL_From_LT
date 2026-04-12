from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template
from dataset import dataset, category2exampleop



@register_prompt("ascendc", "selected_shot")
class AscendcSelectPromptStrategy(BasePromptStrategy):        
    def generate(self, op):
        category = dataset[op]['category']
        # 仅部分 category 在 category2exampleop 中有专用示例；其余与 add_shot 一致回退到 add
        example_op = category2exampleop.get(category, "add")
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('ascendc', op, example_op)
        return ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, example_op)

