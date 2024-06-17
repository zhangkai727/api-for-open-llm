from functools import partial
from typing import Iterator

import anyio
from fastapi import (
    APIRouter,
    Depends,
    Request,
    HTTPException,
    status,
)
from loguru import logger
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.core.default import DefaultEngine
from api.models import LLM_ENGINE
from api.utils.compat import dictify
from api.utils.protocol import CompletionCreateParams
from api.utils.request import (
    handle_request,
    check_api_key,
    get_event_publisher,
)

completion_router = APIRouter()


# 定义生成器函数，用于生成语言模型引擎实例
def get_engine():
    yield LLM_ENGINE

# 创建处理聊天完成请求的POST端点
@chat_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],  # 设置依赖项，用于检查API密钥
    status_code=status.HTTP_200_OK,  # 设置HTTP状态码为成功的200 OK
)
async def create_chat_completion(
    request: ChatCompletionCreateParams,  # 定义接收请求参数的类型
    raw_request: Request,  # 定义原始请求对象的类型
    engine: DefaultEngine = Depends(get_engine),  # 注入语言模型引擎的依赖
):
    """创建聊天完成的响应"""
    # 检查请求是否无效（没有消息或最后一条消息来自助手）
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="无效的请求")

    # 异步处理请求，如果未提供max_tokens，则设置为默认值1024
    request = await handle_request(request, engine.stop)
    request.max_tokens = request.max_tokens or 1024

    # 准备用于引擎完成函数的参数
    params = dictify(request, exclude={"messages"})
    params.update(dict(prompt_or_messages=request.messages, echo=False))
    logger.debug(f"==== 请求详情 ====\n{params}")

    # 在线程池中运行引擎的完成函数
    iterator_or_completion = await run_in_threadpool(engine.create_chat_completion, params)

    # 如果结果是一个迭代器，则流式返回响应
    if isinstance(iterator_or_completion, Iterator):
        # 从迭代器中获取第一个响应
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # 定义一个迭代器函数来逐步生成响应
        def iterator() -> Iterator:
            yield first_response
            yield from iterator_or_completion

        # 创建内存对象流，用于发送和接收数据
        send_chan, recv_chan = anyio.create_memory_object_stream(10)

        # 返回EventSourceResponse以流式返回响应
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator(),  # 提供迭代器以流式返回响应
            ),
        )
    else:
        # 如果不是迭代器，则直接返回完成的响应
        return iterator_or_completion
