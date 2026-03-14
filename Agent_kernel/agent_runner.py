from dataclasses import dataclass
from typing import Optional

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

    return AgentResult(op=task.op, raw_answer=raw_answer)

