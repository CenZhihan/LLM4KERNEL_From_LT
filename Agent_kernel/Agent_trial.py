# 这是一个langgraph的demo

# 查询轮数上限（KB + Web 总轮数）
MAX_QUERY_ROUNDS = 3
# 是否允许使用知识库查询 / 网页搜索（两者可独立开关）
ENABLE_KB_QUERY = True
ENABLE_WEB_SEARCH = True

from typing import TypedDict, List, Annotated

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired  # Python < 3.11

from langgraph.graph import StateGraph, END
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

try:
    from ddgs import DDGS
    _SEARCH_BACKEND = "ddgs"
except ImportError:
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*")
        from duckduckgo_search import DDGS
        _SEARCH_BACKEND = "duckduckgo_search"
    except ImportError:
        DDGS = None
        _SEARCH_BACKEND = None

# 多次搜索结果叠加：新结果追加到已有列表
def add_search_results(left: List[str], right: List[str]) -> List[str]:
    return (left or []) + (right or [])

# 1. 定义 State（推荐继承 MessagesState）
class AgentState(MessagesState):
    search_results: Annotated[List[str], add_search_results]  # 多次网页搜索叠加
    kb_results: Annotated[List[str], add_search_results]  # 多次知识库查询叠加
    query_round_count: NotRequired[int]  # 已执行查询轮数（KB+Web 合计），用于 MAX_QUERY_ROUNDS
    next_action: NotRequired[str]  # choose_tool 写入：KB / WEB / ANSWER
    current_query: NotRequired[str]  # 本轮查询内容（KB 用英文，WEB 用搜索句）
    need_more_search: NotRequired[bool]  # 人类不满意后是否再进入 choose_tool
    human_satisfied: NotRequired[bool]  # 人类介入：是否满意，由 human_review 写入

# 2. 定义模型（使用中转 API）
_xi_api_key = os.getenv("XI_AI_API_KEY")
if not _xi_api_key or not _xi_api_key.strip():
    raise SystemExit(
        "未设置或为空的环境变量 XI_AI_API_KEY。请先设置后再运行，例如：\n"
        "  export XI_AI_API_KEY=你的密钥\n"
        "或在 .env 中配置（若使用 python-dotenv）。"
    )
llm = ChatOpenAI(
    model="gpt-5",                  # 请根据中转服务支持的模型名调整
    api_key=_xi_api_key,
    base_url="https://api-2.xi-ai.cn/v1",  # 中转服务地址
)

# 3. 定义节点函数

def choose_tool_node(state: AgentState) -> dict:
    """根据用户问题与当前已累加结果，让模型选择本回合动作：KB / WEB / ANSWER。"""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else state["messages"][-1].content
    kb_results = state.get("kb_results", [])
    search_results = state.get("search_results", [])
    round_count = state.get("query_round_count", 0)
    at_max = round_count >= MAX_QUERY_ROUNDS

    tools_desc = []
    if ENABLE_KB_QUERY:
        tools_desc.append("知识库查询（KB）")
    if ENABLE_WEB_SEARCH:
        tools_desc.append("网页搜索（WEB）")
    if not tools_desc:
        return {"next_action": "ANSWER", "current_query": ""}

    tools_line = "、".join(tools_desc)
    if ENABLE_KB_QUERY and ENABLE_WEB_SEARCH:
        hint = "建议优先使用知识库。"
    else:
        hint = ""
    kb_rule = "查知识库时，你必须用英文写出查询句（第二行）。" if ENABLE_KB_QUERY else ""

    existing = ""
    if kb_results or search_results:
        if kb_results:
            existing += "已查知识库结果（节选）：\n" + "\n".join(kb_results[:3]) + "\n\n"
        if search_results:
            existing += "已搜网页结果（节选）：\n" + "\n".join(search_results[:3]) + "\n\n"

    prompt = (
        f"用户问题：\n{user_question}\n\n"
        f"当前你可选工具：{tools_line}，或直接回答（ANSWER）。{hint}\n"
        f"规则：只输出第一行动作，取值为 KB、WEB、ANSWER 之一。若选 KB 或 WEB，第二行写出本轮的查询内容（KB 必须英文）。{kb_rule}\n"
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

    if at_max:
        next_action = "ANSWER"
        current_query = ""
    elif "ANSWER" in first or first == "ANSWER":
        next_action = "ANSWER"
        current_query = ""
    elif "KB" in first and ENABLE_KB_QUERY:
        next_action = "KB"
        current_query = query_line or user_question
    elif "WEB" in first and ENABLE_WEB_SEARCH:
        next_action = "WEB"
        current_query = query_line or user_question
    elif ENABLE_KB_QUERY:
        next_action = "KB"
        current_query = query_line or user_question
    elif ENABLE_WEB_SEARCH:
        next_action = "WEB"
        current_query = query_line or user_question
    else:
        next_action = "ANSWER"
        current_query = ""

    print(f"[choose_tool] 模型回复: {raw[:200]!r} -> next_action={next_action}")
    return {"next_action": next_action, "current_query": current_query}


def _ensure_english_for_kb(user_question: str) -> str:
    """若用户问题非英文，用 LLM 翻译为英文查询句。"""
    prompt = (
        f"将下面用户问题转成一句英文查询（用于知识库检索），只输出这句英文，不要解释、不要引号。\n\n{user_question}"
    )
    resp = llm.invoke([{"role": "user", "content": prompt}])
    out = (resp.content or "").strip().strip('"\'')
    return out or user_question


def kb_query_node(state: AgentState) -> dict:
    """执行知识库查询，结果追加到 kb_results，轮数+1。"""
    from db_related.knowledge_query import query_knowledge

    query = state.get("current_query") or ""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else state["messages"][-1].content
    if not query:
        query = user_question
    # 知识库要求英文查询
    if not query.strip().replace(" ", "").isascii():
        query = _ensure_english_for_kb(query)
    chunks = query_knowledge(query, top_k=3)
    round_num = state.get("query_round_count", 0) + 1
    print(f"[Round {round_num}] 工具=知识库(KB), 查询=\"{query}\"")
    return {"kb_results": chunks, "query_round_count": round_num}

def _generate_search_query(state: AgentState) -> str:
    """根据用户需求描述总结出本轮的搜索问题；多轮时根据已有结果生成不同的补充问题。"""
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
    return query or user_need


def _run_web_search(query: str, max_results: int = 5) -> List[str]:
    """使用 ddgs（或 duckduckgo_search）执行真实网页搜索，返回摘要列表。"""
    if DDGS is None:
        return [f"[请安装搜索包: pip install ddgs] 查询: {query}"]
    try:
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results)
            results = list(raw) if raw else []
        out = []
        for r in results:
            if not isinstance(r, dict):
                continue
            title = r.get("title") or r.get("name") or ""
            body = r.get("body") or r.get("snippet") or r.get("description") or ""
            if title or body:
                out.append(f"【{title}】 {body}".strip())
        return out if out else [f"[未返回结果，请稍后重试] 查询: {query}"]
    except Exception as e:
        return [f"[搜索异常: {e}] 查询: {query}"]


def search_node(state: AgentState) -> dict:
    """使用 current_query 或生成搜索句，执行网页搜索并叠加结果，query_round_count+1。"""
    query = state.get("current_query") or _generate_search_query(state)
    results = _run_web_search(query)
    if not results:
        results = ["(未找到相关结果)"]
    round_num = state.get("query_round_count", 0) + 1
    print(f"[Round {round_num}] 工具=网页搜索(WEB), 查询=\"{query}\"")
    return {
        "search_results": results,
        "query_round_count": round_num,
    }

def answer_node(state: AgentState) -> dict:
    kb_results = state.get("kb_results", [])
    search_results = state.get("search_results", [])
    kb_text = "\n".join(kb_results) if kb_results else ""
    search_text = "\n".join(search_results) if search_results else ""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else state["messages"][-1].content

    parts = []
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


# 人类介入：先展示当前回答，再问满意/不满意
def human_review_node(state: AgentState) -> dict:
    """answer 后执行：先打印当前 LLM 回答，再阻塞等待用户输入。"""
    if state["messages"]:
        last_msg = state["messages"][-1]
        content = getattr(last_msg, "content", str(last_msg)) or "(无文本)"
        print("\n========== 当前回答 ==========")
        print(content)
        print("==============================\n")
    print("--- 人类介入 ---")
    print("请输入你的反馈：")
    print("  输入 满意 / y / 是 → 结束")
    print("  输入 不满意 / n / 不对 / 再搜 等 → 将根据当前答案与搜索结果重新判断是否再搜并改进答案")
    raw = input("你的输入: ").strip().lower()
    satisfied = raw in ("满意", "y", "yes", "是", "")
    return {"human_satisfied": satisfied}


# 人类表示不满意后：判断是否再进入 choose_tool 做补充查询（KB/Web）
def re_evaluate_after_human_node(state: AgentState) -> dict:
    """人类不满意时：若未达轮数上限且允许查询，则判断是否再进入选择工具（KB/Web）补充。"""
    rounds = state.get("query_round_count", 0)
    if rounds >= MAX_QUERY_ROUNDS or (not ENABLE_KB_QUERY and not ENABLE_WEB_SEARCH):
        return {"need_more_search": False}

    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else ""
    last_content = state["messages"][-1].content if state["messages"] else ""
    kb_results = state.get("kb_results", [])
    search_results = state.get("search_results", [])
    ref_text = "\n".join(kb_results[:2] + search_results[:2]) if (kb_results or search_results) else "（无）"

    prompt = (
        "用户问题：\n" + user_question + "\n\n当前助手回答：\n" + last_content + "\n\n当前已检索内容（节选）：\n" + ref_text + "\n\n"
        "用户表示不满意。是否需要再查一次知识库或网页以改进回答？若需要则回复 YES，否则回复 NO。只回复 YES 或 NO。"
    )
    resp = llm.invoke([{"role": "user", "content": prompt}])
    content = (resp.content or "").strip().upper()
    need_more = "YES" in content
    return {"need_more_search": need_more}

# 4. 构建图
def entry_node(state: AgentState) -> dict:
    """入口占位，不修改 state。"""
    return {}

def route_entry(state: AgentState) -> str:
    """入口后：若两个查询都关闭则直接回答，否则进入选择工具。"""
    if not ENABLE_KB_QUERY and not ENABLE_WEB_SEARCH:
        return "answer"
    return "choose_tool"

def route_after_choose_tool(state: AgentState) -> str:
    """choose_tool 后：根据 next_action 与轮数上限路由。"""
    action = (state.get("next_action") or "").upper()
    rounds = state.get("query_round_count", 0)
    if action == "ANSWER" or rounds >= MAX_QUERY_ROUNDS:
        return "answer"
    if action == "KB" and ENABLE_KB_QUERY:
        return "kb_query"
    if action == "WEB" and ENABLE_WEB_SEARCH:
        return "search"
    return "answer"

def route_after_human(state: AgentState) -> str:
    """人类介入后：满意则结束，不满意则重新判断是否再查。"""
    return "end" if state.get("human_satisfied") else "re_evaluate"

def route_after_re_evaluate(state: AgentState) -> str:
    """人类不满意且重新判断后：再查则进入 choose_tool，否则结束。"""
    return "choose_tool" if state.get("need_more_search") else "end"

workflow = StateGraph(AgentState)
workflow.add_node("entry", entry_node)
workflow.add_node("choose_tool", choose_tool_node)
workflow.add_node("kb_query", kb_query_node)
workflow.add_node("search", search_node)
workflow.add_node("answer", answer_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("re_evaluate", re_evaluate_after_human_node)

workflow.set_entry_point("entry")
workflow.add_conditional_edges("entry", route_entry, {"answer": "answer", "choose_tool": "choose_tool"})
workflow.add_conditional_edges("choose_tool", route_after_choose_tool, {"answer": "answer", "kb_query": "kb_query", "search": "search"})
workflow.add_edge("kb_query", "choose_tool")
workflow.add_edge("search", "choose_tool")
workflow.add_edge("answer", "human_review")
workflow.add_conditional_edges("human_review", route_after_human, {"end": END, "re_evaluate": "re_evaluate"})
workflow.add_conditional_edges("re_evaluate", route_after_re_evaluate, {"choose_tool": "choose_tool", "end": END})

app = workflow.compile()

# 5. 调用
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    try:
        import openai
    except ImportError:
        openai = None

    # 使用更容易触发搜索的问题（实时/事实类）；若用「我是谁」等身份问题，模型常判为不需搜索
    initial_state = {
        "messages": [HumanMessage(content="Ascend C的softmax怎么写？")]
    }
    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        if openai and isinstance(e, openai.AuthenticationError):
            print("API 认证失败（401）。请检查：")
            print("  1. 环境变量 XI_AI_API_KEY 是否已正确设置：echo $XI_AI_API_KEY")
            print("  2. 密钥是否有效、未过期，且与 base_url 对应的中转服务一致")
            print("  3. 若在 IDE 中运行，需在运行配置里添加 XI_AI_API_KEY 或先 export 再启动")
            raise SystemExit(1) from e
        raise

    # 到达 END 后：先按顺序输出知识库返回（每段约 200 token），再输出网页搜索返回
    MAX_TOKEN_PER_SEGMENT = 200
    _approx_chars = MAX_TOKEN_PER_SEGMENT * 4  # 约 4 字符/token

    print("\n" + "=" * 60)
    print("知识库返回（按顺序，每段最多约 200 token）")
    print("=" * 60)
    for i, r in enumerate(final_state.get("kb_results", [])):
        seg = r if len(r) <= _approx_chars else r[:_approx_chars] + "..."
        print(f"[{i}] {seg}\n")
    print("=" * 60)
    print("网页搜索返回")
    print("=" * 60)
    for i, r in enumerate(final_state.get("search_results", [])):
        print(f"[{i}] {r}\n")
    print("=" * 60)
    print("state['messages'] 全部内容")
    print("=" * 60)
    for i, msg in enumerate(final_state["messages"]):
        role = type(msg).__name__
        content = getattr(msg, "content", str(msg)) or "(无文本)"
        print(f"[{i}] {role}:\n{content}\n")
    print("=================================================")
