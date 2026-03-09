# add_shot_with_doc: same as add_shot but prepend Ascend C API reference from file
from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from prompt_generators.prompt_utils import read_relavant_files, ascendc_template
from config import ascendc_api_reference_path
import os

# Max chars for API doc to avoid exceeding model context; truncate and log if over
ASCENDC_API_DOC_MAX_CHARS = 12000


def _load_api_reference():
    if not ascendc_api_reference_path or not os.path.isfile(ascendc_api_reference_path):
        return ""
    try:
        with open(ascendc_api_reference_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return ""
    if len(content) > ASCENDC_API_DOC_MAX_CHARS:
        content = content[:ASCENDC_API_DOC_MAX_CHARS] + "\n\n(... 已截断，超过长度限制 ...)"
    return content.strip()


@register_prompt("ascendc", "add_shot_with_doc")
class AscendcAddShotWithDocPromptStrategy(BasePromptStrategy):
    def generate(self, op):
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files("ascendc", op, "add")
        base_prompt = ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, "add")
        doc_content = _load_api_reference()
        if not doc_content:
            return base_prompt
        doc_section = "## Ascend C API 参考\n\n" + doc_content + "\n\n---\n\n"
        return doc_section + base_prompt
