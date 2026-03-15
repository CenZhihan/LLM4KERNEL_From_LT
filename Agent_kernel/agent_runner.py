from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from prompt_generators.prompt_registry import PROMPT_REGISTRY
from .agent_config import AgentToolMode
from .agent_builder import build_agent_app, AgentKernelState


@dataclass
class KernelTask:
    language: str
    op: str
    strategy_name: str


@dataclass
class AgentResult:
    op: str
    raw_answer: str
    reasoning: Optional[str] = None
    tool_usage: Optional[List[Dict[str, Any]]] = None


def _build_prompt(language: str, strategy_name: str, op: str) -> str:
    if language not in PROMPT_REGISTRY or strategy_name not in PROMPT_REGISTRY[language]:
        from importlib import import_module

        import_module(f"prompt_generators.{language}_{strategy_name}")
    strategy = PROMPT_REGISTRY[language][strategy_name]
    return strategy.generate(op)


def generate_kernel_with_agent(
    task: KernelTask,
    tool_mode: AgentToolMode,
) -> AgentResult:
    prompt = _build_prompt(task.language, task.strategy_name, task.op)
    # 启用知识库时，在任务描述前鼓励先查知识库再作答，避免模型过于自信直接回答
    if tool_mode in (AgentToolMode.KB_ONLY, AgentToolMode.KB_AND_WEB):
        prompt = (
            "【说明】请先使用知识库检索与本题相关的文档、API 说明或示例，再基于检索结果作答；"
            "不要仅凭已有知识直接写代码。\n\n"
            + prompt
        )

    app = build_agent_app(tool_mode)
    initial_state: AgentKernelState = {
        "messages": [
            HumanMessage(
                content=prompt,
            )
        ]
    }
    final_state = app.invoke(initial_state)
    messages = final_state.get("messages", [])
    raw_answer = ""
    if messages:
        last = messages[-1]
        raw_answer = getattr(last, "content", "") or ""

    tool_calls = final_state.get("tool_calls_log", [])
    tool_usage = tool_calls if tool_calls else None
    return AgentResult(op=task.op, raw_answer=raw_answer, tool_usage=tool_usage)

