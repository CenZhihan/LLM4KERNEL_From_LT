from openai import OpenAI
 
api_key = "sk-lKKkGpPk8a6f2fZp3c50644805Ad4b389f2cE978B39340C9"
api_base = "https://api-2.xi-ai.cn/v1"
client = OpenAI(api_key=api_key, base_url=api_base)
 
completion = client.chat.completions.create(
  model="gpt-5-mini",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!, who are you?"}
  ]
)
 
print(completion.choices[0].message)