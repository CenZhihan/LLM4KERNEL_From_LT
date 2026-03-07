# API 配置文件模板
# 复制此文件为 api_config.py 并填入你的 key / base_url / 模型名：
#   cp api_config.example.py api_config.py
# 使用方式：设置环境变量 USE_API_CONFIG=1 时，脚本会从此文件读取配置；
# 未设置时则从环境变量 OPENAI_API_KEY、OPENAI_API_BASE 等读取。

OPENAI_API_KEY = ""
OPENAI_API_BASE = ""
MODEL = "gpt-5"
