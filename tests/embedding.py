from openai import OpenAI

<<<<<<< HEAD
# 创建一个 OpenAI 客户端实例，用于与 OpenAI API 进行通信
client = OpenAI(
    api_key="EMPTY",  # API 密钥，用于身份验证
    base_url="http://192.168.20.159:8000/v1/",  # OpenAI API 的URL 地址
)

# 调用 embeddings.create 方法来计算文本的嵌入表示
embedding = client.embeddings.create(
    input="你好",  # 要计算嵌入的文本输入
    model="aspire/acge_text_embedding",  # 使用的嵌入模型名称
    dimensions=384,  # 嵌入向量的维度
)

# 打印嵌入向量的长度，此处假设返回的 embedding 对象具有 data 属性，其中包含嵌入向量的信息
print(len(embedding.data[0].embedding))

=======
client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.20.159:8000/v1/",
)


# compute the embedding of the text
embedding = client.embeddings.create(
    input="你好",
    model="aspire/acge_text_embedding",
    dimensions=384,
)
print(len(embedding.data[0].embedding))
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
