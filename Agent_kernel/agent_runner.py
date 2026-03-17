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
    report: Optional[Dict[str, Any]] = None


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
    # 启用知识库时，在任务描述前提示可以利用 AscendC API 文档，降低编造 API 的概率
    if tool_mode in (AgentToolMode.KB_ONLY, AgentToolMode.KB_AND_WEB):
        prompt = (
            "[Note] You have access to a knowledge base that contains Huawei Ascend C API documentation. "
            "This documentation is highly reliable; following it when writing kernels can greatly reduce the chance of inventing non-existent APIs. "
            "You are strongly encouraged to consult the knowledge base before answering, but it is not strictly required.\n\n"
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

    reasoning = (final_state.get("reasoning_content") or "").strip() or None
    tool_calls = final_state.get("tool_calls_log", []) or []
    report = {
        "reasoning_content": reasoning,
        "answer": raw_answer,
        "tool_calls": [
            {
                "round": t.get("round"),
                "tool": t.get("tool", ""),
                "query": t.get("query", ""),
                "response": t.get("response", ""),
            }
            for t in tool_calls
        ],
    }
    return AgentResult(
        op=task.op,
        raw_answer=raw_answer,
        reasoning=reasoning,
        tool_usage=tool_calls if tool_calls else None,
        report=report,
    )

