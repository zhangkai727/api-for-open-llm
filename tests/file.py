<<<<<<< HEAD
from openai import OpenAI
import requests

# 创建一个 OpenAI 客户端实例，用于与 OpenAI API 进行通信
client = OpenAI(
    api_key="EMPTY",  # API 密钥，用于身份验证
    base_url="http://192.168.20.159:8000/v1",  # OpenAI API 的URL 地址
)

# 调用 client 的 files.list 方法，列出当前用户的文件列表，并打印出来
print(client.files.list())

# 使用 client 的 files.create 方法，上传文件到 OpenAI 服务端，并指定文件的用途为 "chat"
uf = client.files.create(
    file=open("../README.md", "rb"),  # 打开要上传的文件对象
    purpose="chat",  # 指定文件用途
)
print(uf)  # 打印上传文件后返回的对象信息

# 使用 client 的 files.delete 方法，删除之前上传的文件，传入文件 ID（uf.id）
df = client.files.delete(file_id=uf.id)
print(df)  # 打印文件删除操作的返回结果

# 发送 POST 请求到指定 URL，将文件进行分块处理，然后将响应转换为 JSON 格式并打印出来
print(
    requests.post(
        url="http://192.168.20.159:8000/v1/files/split",  # 目标 URL
        json={},  # 发送的 JSON 数据为空对象
        files={"file": open("../README.md", "rb")}  # 上传的文件对象
    ).json()
)

=======
import requests
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.0.59:7891/v1/",
)

print(client.files.list())


uf = client.files.create(
    file=open("../README.md", "rb"),
    purpose="chat",
)
print(uf)


print(
    requests.post(
        url="http://192.168.0.59:7891/v1/files/split",
        json={"file_id": uf.id},
    ).json()
)

df = client.files.delete(file_id=uf.id)
print(df)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
