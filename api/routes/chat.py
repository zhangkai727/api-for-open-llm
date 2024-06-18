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

<<<<<<< HEAD
from api.core.default import DefaultEngine
from api.models import LLM_ENGINE
from api.utils.compat import dictify
from api.utils.protocol import ChatCompletionCreateParams, Role
from api.utils.request import (
    handle_request,
=======
from api.common import dictify
from api.engine.hf import HuggingFaceEngine
from api.models import LLM_ENGINE
from api.protocol import ChatCompletionCreateParams, Role
from api.utils import (
    check_completion_requests,
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    check_api_key,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")

<<<<<<< HEAD
# 定义生成器函数，生成语言模型引擎实例
def get_engine():
    yield LLM_ENGINE

# 创建处理聊天完成请求的POST端点
@chat_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],  # 依赖项，用于检查API密钥
    status_code=status.HTTP_200_OK,  # 成功的HTTP状态码
)
async def create_chat_completion(
    request: ChatCompletionCreateParams,  # 接收的请求参数
    raw_request: Request,  # 原始的请求对象
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
=======

def get_engine():
    yield LLM_ENGINE


@chat_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine: HuggingFaceEngine = Depends(get_engine),
):
    """Creates a completion for the chat message"""
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT.value:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await check_completion_requests(
        request,
        engine.template.stop,
        engine.template.stop_token_ids,
    )
    request.max_tokens = request.max_tokens or 1024

    params = dictify(request, exclude={"messages"})
    params.update(dict(prompt_or_messages=request.messages, echo=False))
    logger.debug(f"==== request ====\n{params}")

    iterator_or_completion = await run_in_threadpool(engine.create_chat_completion, params)

    if isinstance(iterator_or_completion, Iterator):
        # It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid, and we can use it to stream the response.
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        def iterator() -> Iterator:
            yield first_response
            yield from iterator_or_completion

<<<<<<< HEAD
        # 创建内存对象流，用于发送和接收数据
        send_chan, recv_chan = anyio.create_memory_object_stream(10)

        # 返回EventSourceResponse以流式返回响应
=======
        send_chan, recv_chan = anyio.create_memory_object_stream(10)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
<<<<<<< HEAD
                iterator=iterator(),  # 提供迭代器以流式返回响应
            ),
        )
    else:
        # 如果不是迭代器，则直接返回完成的响应
        return iterator_or_completion

=======
                iterator=iterator(),
            ),
        )
    else:
        return iterator_or_completion
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
