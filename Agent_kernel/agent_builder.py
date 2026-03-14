from typing import List, Annotated

from typing_extensions import NotRequired

from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .agent_config import AgentToolMode, get_llm_config_from_env

try:
    from ddgs import DDGS  # type: ignore
    _SEARCH_BACKEND = "ddgs"
except ImportError:  # pragma: no cover - optional dependency
    try:
        import warnings

        warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*")
        from duckduckgo_search import DDGS  # type: ignore

        _SEARCH_BACKEND = "duckduckgo_search"
    except ImportError:  # pragma: no cover - optional dependency
        DDGS = None  # type: ignore
        _SEARCH_BACKEND = None


MAX_QUERY_ROUNDS = 3


def _add_list(left: List[str], right: List[str]) -> List[str]:
    return (left or []) + (right or [])


class AgentKernelState(MessagesState):
    search_results: Annotated[List[str], _add_list]
    kb_results: Annotated[List[str], _add_list]
    query_round_count: NotRequired[int]
    next_action: NotRequired[str]
    current_query: NotRequired[str]


def _build_llm():
    cfg = get_llm_config_from_env()
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )


def _ensure_english_for_kb(llm, user_question: str) -> str:
    prompt = (
        "将下面用户问题转成一句英文查询（用于知识库检索），"
        "只输出这句英文，不要解释、不要引号。\n\n"
        f"{user_question}"
    )
    resp = llm.invoke([{"role": "user", "content": prompt}])
    out = (resp.content or "").strip().strip('"\'')
    return out or user_question


def _run_web_search(query: str, max_results: int = 5) -> List[str]:
    if DDGS is None:
        return [f"[请安装搜索包: pip install ddgs] 查询: {query}"]
    try:
        with DDGS() as ddgs:  # type: ignore
            raw = ddgs.text(query, max_results=max_results)
            results = list(raw) if raw else []
        out: List[str] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            title = r.get("title") or r.get("name") or ""
            body = r.get("body") or r.get("snippet") or r.get("description") or ""
            if title or body:
                out.append(f"【{title}】 {body}".strip())
        return out if out else [f"[未返回结果，请稍后重试] 查询: {query}"]
    except Exception as e:  # pragma: no cover - 网络异常
        return [f"[搜索异常: {e}] 查询: {query}"]


def build_agent_app(tool_mode: AgentToolMode):
    llm = _build_llm()

    enable_kb = tool_mode in (AgentToolMode.KB_ONLY, AgentToolMode.KB_AND_WEB)
    enable_web = tool_mode in (AgentToolMode.WEB_ONLY, AgentToolMode.KB_AND_WEB)

    def choose_tool_node(state: AgentKernelState) -> dict:
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_question = user_msgs[0].content if user_msgs else state["messages"][-1].content
        kb_results = state.get("kb_results", [])
        search_results = state.get("search_results", [])
        round_count = state.get("query_round_count", 0)
        at_max = round_count >= MAX_QUERY_ROUNDS

        tools_desc: List[str] = []
        if enable_kb:
            tools_desc.append("知识库查询（KB）")
        if enable_web:
            tools_desc.append("网页搜索（WEB）")
        if not tools_desc or tool_mode == AgentToolMode.NO_TOOL:
            return {"next_action": "ANSWER", "current_query": ""}

        tools_line = "、".join(tools_desc)
        if enable_kb and enable_web:
            hint = "建议优先使用知识库。"
        else:
            hint = ""
        kb_rule = "查知识库时，你必须用英文写出查询句（第二行）。" if enable_kb else ""

        existing = ""
        if kb_results or search_results:
            if kb_results:
                existing += "已查知识库结果（节选）：\n" + "\n".join(kb_results[:3]) + "\n\n"
            if search_results:
                existing += "已搜网页结果（节选）：\n" + "\n".join(search_results[:3]) + "\n\n"

        prompt = (
            f"用户问题：\n{user_question}\n\n"
            f"当前你可选工具：{tools_line}，或直接回答（ANSWER）。{hint}\n"
            "规则：只输出第一行动作，取值为 KB、WEB、ANSWER 之一。"
            "若选 KB 或 WEB，第二行写出本轮的查询内容（KB 必须英文）。"
            f"{kb_rule}\n"
        )
        if existing:
            prompt += f"已有检索内容：\n{existing}\n"
        if at_max:
            prompt += f"已达最大查询轮数（{MAX_QUERY_ROUNDS}），本回合只能选 ANSWER。\n"
        prompt += "第一行只输出：KB 或 WEB 或 ANSWER。"

        resp = llm.invoke([{"role": "user", "content": prompt}])
        raw = (resp.content or "").strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        first = (lines[0].upper() if lines else "") or "ANSWER"
        query_line = lines[1] if len(lines) > 1 else ""

        if at_max or tool_mode == AgentToolMode.NO_TOOL:
            next_action = "ANSWER"
            current_query = ""
        elif "ANSWER" in first or first == "ANSWER":
            next_action = "ANSWER"
            current_query = ""
        elif "KB" in first and enable_kb:
            next_action = "KB"
            current_query = query_line or user_question
        elif "WEB" in first and enable_web:
            next_action = "WEB"
            current_query = query_line or user_question
        elif enable_kb:
            next_action = "KB"
            current_query = query_line or user_question
        elif enable_web:
            next_action = "WEB"
            current_query = query_line or user_question
        else:
            next_action = "ANSWER"
            current_query = ""

        return {"next_action": next_action, "current_query": current_query}

    def kb_query_node(state: AgentKernelState) -> dict:
        from .db_related.knowledge_query import query_knowledge

        query = state.get("current_query") or ""
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_question = user_msgs[0].content if user_msgs else state["messages"][-1].content
        if not query:
            query = user_question
        if not query.strip().replace(" ", "").isascii():
            query = _ensure_english_for_kb(llm, query)
        chunks = query_knowledge(query, top_k=3)
        round_num = state.get("query_round_count", 0) + 1
        return {"kb_results": chunks, "query_round_count": round_num}

    def search_node(state: AgentKernelState) -> dict:
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_need = user_msgs[0].content if user_msgs else state["messages"][-1].content
        existing = state.get("search_results", [])
        existing_text = "\n".join(existing) if existing else ""
        if not existing_text:
            prompt = (
                f"用户需求描述：\n{user_need}\n\n"
                "请用一句话总结成一个适合在搜索引擎中查询的问题（只输出这一句话，不要解释、不要引号）。"
            )
        else:
            prompt = (
                f"用户需求：\n{user_need}\n\n目前已搜到的信息：\n{existing_text}\n\n"
                "请再提出一个与上述不同的、能补充获取更多相关信息的搜索问题（一句话，只输出这句话，不要解释）。"
            )
        resp = llm.invoke([{"role": "user", "content": prompt}])
        query = (resp.content or "").strip().strip('"\'')
        query = query or user_need
        results = _run_web_search(query)
        if not results:
            results = ["(未找到相关结果)"]
        round_num = state.get("query_round_count", 0) + 1
        return {"search_results": results, "query_round_count": round_num}

    def answer_node(state: AgentKernelState) -> dict:
        kb_results = state.get("kb_results", [])
        search_results = state.get("search_results", [])
        kb_text = "\n".join(kb_results) if kb_results else ""
        search_text = "\n".join(search_results) if search_results else ""
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        user_question = user_msgs[0].content if user_msgs else state["messages"][-1].content

        parts: List[str] = []
        if kb_text:
            parts.append("【知识库检索结果（英文）】\n" + kb_text)
        if search_text:
            parts.append("【网页搜索结果】\n" + search_text)
        ref_text = "\n\n".join(parts) if parts else ""

        if ref_text:
            system_prompt = (
                "你是一个助手。下面是根据用户问题检索到的参考资料（知识库为英文、网页为搜索摘要）。"
                "请基于这些内容用**中文**回答用户问题。\n\n"
                f"参考资料：\n{ref_text}\n\n"
                f"用户问题：{user_question}\n\n"
                "请用中文给出完整、清晰的回答。"
            )
        else:
            system_prompt = (
                "你是一个助手。当前未进行知识库或网页查询，请仅根据你的知识用**中文**回答用户问题。\n\n"
                f"用户问题：{user_question}"
            )

        resp = llm.invoke([{"role": "system", "content": system_prompt}])
        return {"messages": [resp]}

    def entry_node(state: AgentKernelState) -> dict:
        return {}

    def route_entry(state: AgentKernelState) -> str:
        if tool_mode == AgentToolMode.NO_TOOL or (not enable_kb and not enable_web):
            return "answer"
        return "choose_tool"

    def route_after_choose_tool(state: AgentKernelState) -> str:
        action = (state.get("next_action") or "").upper()
        rounds = state.get("query_round_count", 0)
        if action == "ANSWER" or rounds >= MAX_QUERY_ROUNDS or tool_mode == AgentToolMode.NO_TOOL:
            return "answer"
        if action == "KB" and enable_kb:
            return "kb_query"
        if action == "WEB" and enable_web:
            return "search"
        return "answer"

    workflow = StateGraph(AgentKernelState)
    workflow.add_node("entry", entry_node)
    workflow.add_node("choose_tool", choose_tool_node)
    if enable_kb:
        workflow.add_node("kb_query", kb_query_node)
    if enable_web:
        workflow.add_node("search", search_node)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("entry")
    workflow.add_conditional_edges(
        "entry", route_entry, {"answer": "answer", "choose_tool": "choose_tool"}
    )
    workflow.add_conditional_edges(
        "choose_tool",
        route_after_choose_tool,
        {
            "answer": "answer",
            "kb_query": "kb_query" if enable_kb else "answer",
            "search": "search" if enable_web else "answer",
        },
    )
    if enable_kb:
        workflow.add_edge("kb_query", "choose_tool")
    if enable_web:
        workflow.add_edge("search", "choose_tool")

    app = workflow.compile()
    return app

