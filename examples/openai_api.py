from typing import Optional

from fastapi import Depends, HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI
from sse_starlette import EventSourceResponse

from api.utils.protocol import ChatCompletionCreateParams, CompletionCreateParams, EmbeddingCreateParams

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEYS = None  # 此处设置允许访问接口的api_key列表
# 此处设置模型和接口地址的对应关系
MODEL_LIST = {
    "chat":
        {
            # 模型名称
            "qwen-7b-chat": {
                "addtional_names": ["gpt-3.5-turbo"],  # 其他允许访问该模型的名称，比如chatgpt-next-web使用gpt-3.5-turbo，则需要加入到此处
                "base_url": "http://192.168.20.59:7891/v1",  # 实际访问该模型的接口地址
                "api_key": "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"
            },
            # 模型名称
            "baichuan2-13b": {
                "addtional_names": [],  # 其他允许访问该模型的名称
                "base_url": "http://192.168.20.44:7860/v1",  # 实际访问该模型的接口地址
                "api_key": "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"
            },
            # 需要增加其他模型，仿照上面的例子添加即可
        },
    "completion":
        {
            "sqlcoder": {
                "addtional_names": [],  # 其他允许访问该模型的名称
                "base_url": "http://192.168.20.59:7892/v1",  # 实际访问该模型的接口地址
                "api_key": "xxx"
            },
            # 需要增加其他模型，仿照上面的例子添加即可
        },
    "embedding":
        {
            "base_url": "http://192.168.20.59:8001/v1",  # 实际访问该模型的接口地址
            "api_key": "xxx",  # api_key
        },
}

# 将 "chat" 模型列表中的每个模型及其附加名称映射到模型名称的字典
CHAT_MODEL_MAP = {am: name for name, detail in MODEL_LIST["chat"].items() for am in (detail["addtional_names"] + [name])}

# 将 "completion" 模型列表中的每个模型及其附加名称映射到模型名称的字典
COMPLETION_MODEL_MAP = {am: name for name, detail in MODEL_LIST["completion"].items() for am in (detail["addtional_names"] + [name])}

async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):
    # 检查 API 密钥是否有效的异步函数
    if API_KEYS is not None:
        # 如果 API_KEYS 不为空，则验证授权信息
        if auth is None or (token := auth.credentials) not in API_KEYS:
            # 如果未提供授权信息或提供的令牌不在 API_KEYS 中，则返回 401 错误
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token  # 返回有效的 API 密钥令牌
    else:
        # 如果 API_KEYS 未设置，则允许所有请求通过
        return None  # 返回空，表示未设置 API 密钥，允许所有请求通过



@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: ChatCompletionCreateParams):
    # 检查请求的模型是否有效
    if request.model not in CHAT_MODEL_MAP:
        raise HTTPException(status_code=404, detail="Invalid model")

    # 获取实际的模型名称
    model = CHAT_MODEL_MAP[request.model]

    # 创建异步 OpenAI 客户端
    client = AsyncOpenAI(
        api_key=MODEL_LIST["chat"][model]["api_key"],
        base_url=MODEL_LIST["chat"][model]["base_url"],
    )

    # 将请求参数转换为字典，包括有效字段，排除空值
    params = request.dict(
        exclude_none=True,
        include={
            "messages",
            "model",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
            "stream",
        }
    )

    # 调用 OpenAI 客户端进行聊天完成请求
    response = await client.chat.completions.create(**params)

    # 定义用于生成聊天完成响应的流生成器
    async def chat_completion_stream_generator():
        async for chunk in response:
            yield chunk.json()
        yield "[DONE]"

    # 如果请求指定为流式传输，则返回事件源响应
    if request.stream:
        return EventSourceResponse(chat_completion_stream_generator())

    # 否则直接返回完整的响应对象
    return response


from fastapi import Depends, HTTPException
from starlette.responses import EventSourceResponse

@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
async def create_completion(request: CompletionCreateParams):
    # 检查请求的模型是否有效
    if request.model not in COMPLETION_MODEL_MAP:
        raise HTTPException(status_code=404, detail="Invalid model")

    # 获取实际的模型名称
    model = COMPLETION_MODEL_MAP[request.model]

    # 创建异步的 OpenAI 客户端
    client = AsyncOpenAI(
        api_key=MODEL_LIST["completion"][model]["api_key"],
        base_url=MODEL_LIST["completion"][model]["base_url"],
    )

    # 将请求参数转换为字典，包括有效字段，排除空值
    params = request.dict(
        exclude_none=True,
        include={
            "prompt",
            "model",
            "best_of",
            "echo",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "suffix",
            "temperature",
            "top_p",
            "user",
            "stream",
        }
    )

    # 调用 OpenAI 客户端进行完成请求
    response = await client.completions.create(**params)

    # 定义用于生成完成响应的流生成器
    async def generate_completion_stream_generator():
        async for chunk in response:
            yield chunk.json()
        yield "[DONE]"

    # 如果请求指定为流式传输，则返回事件源响应
    if request.stream:
        return EventSourceResponse(generate_completion_stream_generator())

    # 否则直接返回完整的响应对象
    return response


@app.post("/v1/embeddings", dependencies=[Depends(check_api_key)])
async def create_embeddings(request: EmbeddingCreateParams):
    # 创建异步的 OpenAI 客户端，使用嵌入模型的 API 密钥和基础 URL
    client = AsyncOpenAI(
        api_key=MODEL_LIST["embedding"]["api_key"],
        base_url=MODEL_LIST["embedding"]["base_url"],
    )

    # 调用 OpenAI 客户端进行嵌入创建请求，传入请求参数并排除空值
    embeddings = await client.embeddings.create(**request.dict(exclude_none=True))

    # 返回嵌入结果
    return embeddings


if __name__ == "__main__":
    import uvicorn

    # 运行 FastAPI 应用程序，指定主机和端口，并设置日志级别为 info
    uvicorn.run(app, host="127.0.0.1", port=9009, log_level="info")

