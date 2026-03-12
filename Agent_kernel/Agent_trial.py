# 这是一个langgraph的demo

# search 完成后立即判断：根据「问题 + 当前搜索结果」是否要再搜一轮
MAX_SEARCH_ROUNDS = 3

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
    search_results: Annotated[List[str], add_search_results]  # 多次搜索叠加存储
    use_search: NotRequired[bool]  # 由 decide 节点写入，用于条件边
    need_more_search: NotRequired[bool]  # search 后 / 人类不满意后 是否再搜
    search_round_count: NotRequired[int]  # 已执行搜索轮数，用于限制最大轮数
    human_satisfied: NotRequired[bool]  # 人类介入：是否满意，由 human_review 写入

# 2. 定义模型（使用中转 API）
llm = ChatOpenAI(
    model="gpt-5",                  # 请根据中转服务支持的模型名调整
    api_key=os.getenv("XI_AI_API_KEY"),
    base_url="https://api-2.xi-ai.cn/v1",  # 中转服务地址
)

# 3. 定义节点函数

def decide_node(state: AgentState) -> dict:
    """根据用户问题判断是否需要调用搜索工具。"""
    user_question = state["messages"][-1].content
    prompt = (
        "判断下面的用户问题是否必须通过搜索（查文档、查资料、查最新信息）才能较好回答。"
        "若仅凭通用知识就能回答则不需要搜索；若涉及实时信息、具体数据、文档内容等，则需要搜索。\n"
        "不确定时请优先选择搜索。只回复 YES 或 NO。\n\n"
        f"用户问题：{user_question}"
    )
    resp = llm.invoke([{"role": "user", "content": prompt}])
    raw = (resp.content or "").strip()
    content_upper = raw.upper()
    # 兼容英文 YES 与中文「需要」「是」等
    use_search = (
        "YES" in content_upper
        or "需要" in raw
        or raw in ("是", "要", "要搜索")
        or raw.startswith("是，") or raw.startswith("是的")
    )
    print(f"[decide] 模型回复: {raw!r} -> use_search={use_search}")
    if use_search:
        return {"use_search": True}
    return {"use_search": False, "search_results": []}

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
    """先由 LLM 根据需求总结出本轮的搜索问题（多轮可不同），再执行真实搜索并叠加结果。"""
    print("[search] 正在调用搜索工具...")
    query = _generate_search_query(state)
    print(f"[search] 本轮查询: {query}")
    results = _run_web_search(query)
    if not results:
        results = ["(未找到相关结果)"]
    round_num = state.get("search_round_count", 0) + 1
    return {
        "search_results": results,
        "search_round_count": round_num,
    }

def answer_node(state: AgentState) -> dict:
    # 可能未经过 search，或多轮搜索已叠加，用 .get 取全部搜索结果
    search_results = state.get("search_results", [])
    search_text = "\n".join(search_results) if search_results else ""
    # 始终针对「第一条用户消息」作答（多轮搜索都是为同一个问题服务）
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else state["messages"][-1].content

    if search_text:
        system_prompt = (
            "你是一个搜索助手。下面是（可能多轮）搜索的汇总结果，请基于这些结果回答用户问题。\n\n"
            f"搜索结果：\n{search_text}\n\n"
            f"用户问题：{user_question}"
        )
    else:
        system_prompt = (
            "你是一个助手。当前未进行搜索，请仅根据你的知识回答用户问题。\n\n"
            f"用户问题：{user_question}"
        )

    resp = llm.invoke([{"role": "system", "content": system_prompt}])
    return {"messages": [resp]}


def check_need_more_after_search_node(state: AgentState) -> dict:
    """search 后执行：根据用户问题与当前已搜到的结果，判断是否再搜一轮。"""
    rounds = state.get("search_round_count", 0)
    if rounds >= MAX_SEARCH_ROUNDS:
        return {"need_more_search": False}

    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else ""
    search_results = state.get("search_results", [])
    search_text = "\n".join(search_results) if search_results else "（无）"

    prompt = (
        "用户问题：\n" + user_question + "\n\n当前已搜到的结果：\n" + search_text + "\n\n"
        "仅根据上述结果是否足以较好回答用户问题？若还缺关键信息需要再搜一轮，回复 YES；若已够用则回复 NO。只回复 YES 或 NO。"
    )
    resp = llm.invoke([{"role": "user", "content": prompt}])
    content = (resp.content or "").strip().upper()
    need_more = "YES" in content
    return {"need_more_search": need_more}


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


# 人类表示不满意后：根据当前答案 + 当前搜索结果，判断是否再搜；若再搜则叠加结果后重新 answer
def re_evaluate_after_human_node(state: AgentState) -> dict:
    """人类不满意时执行：根据当前回答与搜索结果，判断是否再搜索一轮以改进答案。"""
    rounds = state.get("search_round_count", 0)
    if rounds >= MAX_SEARCH_ROUNDS:
        return {"need_more_search": False}

    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_question = user_msgs[0].content if user_msgs else ""
    last_content = state["messages"][-1].content if state["messages"] else ""
    search_results = state.get("search_results", [])
    search_text = "\n".join(search_results) if search_results else "（无）"

    prompt = (
        "用户问题：\n" + user_question + "\n\n当前助手回答：\n" + last_content + "\n\n当前搜索结果：\n" + search_text + "\n\n"
        "用户表示不满意。是否需要再搜索一次以获取更多信息来改进回答？若需要则回复 YES，否则回复 NO。只回复 YES 或 NO。"
    )
    resp = llm.invoke([{"role": "user", "content": prompt}])
    content = (resp.content or "").strip().upper()
    need_more = "YES" in content
    return {"need_more_search": need_more}

# 4. 构建图
def route_after_decide(state: AgentState) -> str:
    """根据是否需要搜索，决定下一节点。"""
    return "do_search" if state.get("use_search") else "no_search"

def route_after_search_check(state: AgentState) -> str:
    """search 后：根据问题+当前结果是否再搜一轮。"""
    return "search" if state.get("need_more_search") else "answer"

def route_after_human(state: AgentState) -> str:
    """人类介入后：满意则结束，不满意则重新判断是否再搜。"""
    return "end" if state.get("human_satisfied") else "re_evaluate"

def route_after_re_evaluate(state: AgentState) -> str:
    """人类不满意且重新判断后：再搜则去 search 并改答案，否则结束。"""
    return "search" if state.get("need_more_search") else "end"

workflow = StateGraph(AgentState)
workflow.add_node("decide", decide_node)
workflow.add_node("search", search_node)
workflow.add_node("answer", answer_node)
workflow.add_node("check_after_search", check_need_more_after_search_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("re_evaluate", re_evaluate_after_human_node)

workflow.set_entry_point("decide")
workflow.add_conditional_edges("decide", route_after_decide, {"do_search": "search", "no_search": "answer"})
workflow.add_edge("search", "check_after_search")
workflow.add_conditional_edges("check_after_search", route_after_search_check, {"search": "search", "answer": "answer"})
workflow.add_edge("answer", "human_review")
workflow.add_conditional_edges("human_review", route_after_human, {"end": END, "re_evaluate": "re_evaluate"})
workflow.add_conditional_edges("re_evaluate", route_after_re_evaluate, {"search": "search", "end": END})

app = workflow.compile()

# 5. 调用
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # 使用更容易触发搜索的问题（实时/事实类）；若用「我是谁」等身份问题，模型常判为不需搜索
    initial_state = {
        "messages": [HumanMessage(content="今天适不适合去北京旅游？")]
    }
    final_state = app.invoke(initial_state)

    # 到达 END 后，打印 state["messages"] 与叠加的搜索结果
    print("\n========== state['search_results']（多轮叠加）==========")
    for i, r in enumerate(final_state.get("search_results", [])):
        print(f"  [{i}] {r}")
    print("========== state['messages'] 全部内容 ==========")
    for i, msg in enumerate(final_state["messages"]):
        role = type(msg).__name__
        content = getattr(msg, "content", str(msg)) or "(无文本)"
        print(f"[{i}] {role}:\n{content}\n")
    print("=================================================")
