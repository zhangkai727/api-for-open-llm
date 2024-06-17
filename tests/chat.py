from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="http://192.168.20.44:7861/v1/",
)








# List models API
models = client.models.list()
print(models.model_dump())


# Chat completion API
# 发起一个同步聊天完成请求，返回聊天完成的结果对象
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办",
        }
    ],
    model="gpt-3.5-turbo",
)

# 打印聊天完成的结果对象
print(chat_completion)

# 发起一个流式聊天完成请求，并逐步处理结果
stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办",
        }
    ],
    model="gpt-3.5-turbo",
    stream=True,
)
# 逐步打印每一部分的聊天完成结果内容
for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)

