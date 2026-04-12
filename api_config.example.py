# API 配置文件模板
# 复制此文件为 api_config.py 并填入你的 key / base_url / 模型名：
#   cp api_config.example.py api_config.py
# 使用方式：设置环境变量 USE_API_CONFIG=1 时，脚本会从此文件读取配置；
# 未设置时则从环境变量 OPENAI_API_KEY、OPENAI_API_BASE 等读取。
#
# 推荐（与 OpenAI 兼容网关一致）：
OPENAI_API_KEY = ""
OPENAI_API_BASE = ""  # 例如 https://api.openai.com/v1 或你的代理 /v1
MODEL = "gpt-5"
#
# 若你从 MultiKernelBench 拷来的文件里只有下面两个变量，也可使用（utils 会回退读取）：
# XI_AI_API_KEY = ""
# XI_AI_BASE_URL = ""
