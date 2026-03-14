from enum import Enum
import os


class AgentToolMode(str, Enum):
    NO_TOOL = "no_tool"
    KB_ONLY = "kb_only"
    WEB_ONLY = "web_only"
    KB_AND_WEB = "kb_and_web"


def get_llm_config_from_env() -> dict:
    api_key = os.getenv("XI_AI_API_KEY")
    if not api_key or not api_key.strip():
        raise SystemExit(
            "未设置或为空的环境变量 XI_AI_API_KEY。请先设置后再运行，例如：\n"
            "  export XI_AI_API_KEY=你的密钥\n"
        )
    base_url = os.getenv("XI_AI_BASE_URL", "https://api-2.xi-ai.cn/v1")
    model_name = os.getenv("XI_AI_MODEL", "gpt-5")
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name,
    }

