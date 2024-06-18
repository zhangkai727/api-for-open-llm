from openai import OpenAI

client = OpenAI(
<<<<<<< HEAD
    api_key="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",
=======
    api_key="EMPTY",
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    base_url="http://192.168.20.44:7861/v1/",
)


<<<<<<< HEAD






=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
# List models API
models = client.models.list()
print(models.model_dump())


# Chat completion API
<<<<<<< HEAD
# 发起一个同步聊天完成请求，返回聊天完成的结果对象
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办",
        }
    ],
    model="gpt-3.5-turbo",
)
<<<<<<< HEAD

# 打印聊天完成的结果对象
print(chat_completion)

# 发起一个流式聊天完成请求，并逐步处理结果
=======
print(chat_completion)


>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
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
<<<<<<< HEAD
# 逐步打印每一部分的聊天完成结果内容
for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)

=======
for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
