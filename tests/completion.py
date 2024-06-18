from openai import OpenAI

<<<<<<< HEAD
# 创建一个 OpenAI 客户端实例，用于与 OpenAI API 进行通信
client = OpenAI(
    api_key="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",  # API 密钥，用于身份验证
    base_url="http://192.168.20.44:7861/v1/",  # OpenAI API 的 URL 地址
)

# 同步方式调用聊天完成 API，并返回聊天完成的结果对象
completion = client.completions.create(
    model="gpt-3.5-turbo",  # 指定使用的模型名称
    prompt="感冒了怎么办",  # 提供的输入提示语句
)
# 打印聊天完成的结果对象
print(completion)

# 发起一个流式聊天完成请求，并逐步处理结果
stream = client.completions.create(
    model="gpt-3.5-turbo",  # 指定使用的模型名称
    prompt="感冒了怎么办",  # 提供的输入提示语句
    stream=True,  # 使用流式处理模式
)
# 逐步打印每一部分的聊天完成结果中的文本内容
for part in stream:
    print(part.choices[0].text or "", end="", flush=True)

=======
client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.20.44:7861/v1/",
)


# Chat completion API
completion = client.completions.create(
    model="gpt-3.5-turbo",
    prompt="感冒了怎么办",
)
print(completion)


stream = client.completions.create(
    model="gpt-3.5-turbo",
    prompt="感冒了怎么办",
    stream=True,
)
for part in stream:
    print(part.choices[0].text or "", end="", flush=True)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
