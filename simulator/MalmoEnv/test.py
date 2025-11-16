from openai import OpenAI
import os
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("openrouter_api_key"),
)

completion = client.chat.completions.create(
  model="deepseek/deepseek-r1-0528-qwen3-8b:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)

print(completion.choices[0].message.content)
