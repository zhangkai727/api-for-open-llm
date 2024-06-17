from functools import partial
from typing import Iterator

import anyio
from fastapi import (
    APIRouter,
    Depends,
    Request,
    status,
)
from loguru import logger
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from api.core.llama_cpp_engine import LlamaCppEngine
from api.llama_cpp_routes.utils import get_llama_cpp_engine
from api.utils.compat import dictify
from api.utils.protocol import CompletionCreateParams
from api.utils.request import (
    handle_request,
    check_api_key,
    get_event_publisher,
)

completion_router = APIRouter()


@completion_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],  # 依赖函数检查 API 密钥
    status_code=status.HTTP_200_OK,  # 返回状态码 200 表示成功
)
async def create_completion(
    request: CompletionCreateParams,  # 请求参数是 CompletionCreateParams 类型
    raw_request: Request,  # 原始请求对象
    engine: LlamaCppEngine = Depends(get_llama_cpp_engine),  # 使用 LlamaCppEngine 实例依赖注入
):
    if isinstance(request.prompt, list):
        assert len(request.prompt) <= 1  # 断言确保 prompt 是列表且长度不超过1
        request.prompt = request.prompt[0] if len(request.prompt) > 0 else ""  # 如果 prompt 是列表，则取第一个元素作为字符串，否则设置为空字符串

    request.max_tokens = request.max_tokens or 256  # 设置最大 tokens 数量，默认为 256
    request = await handle_request(request, engine.stop)  # 处理请求，使用引擎的 stop 方法

    include = {
        "temperature",
        "top_p",
        "stream",
        "stop",
        "model",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
    }
    kwargs = dictify(request, include=include)  # 使用 dictify 函数从请求中构造 kwargs 参数字典
    logger.debug(f"==== request ====\n{kwargs}")  # 记录调试日志，输出请求参数

    iterator_or_completion = await run_in_threadpool(
        engine.create_completion, **kwargs  # 在线程池中运行引擎的 create_completion 方法
    )

    if isinstance(iterator_or_completion, Iterator):
        # 若 iterator_or_completion 是迭代器对象，则执行以下逻辑
        # It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)  # 获取迭代器的第一个响应

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid, and we can use it to stream the response.
        # 如果 first_response 没有引发异常，我们可以假定迭代器是有效的，并且可以用它来流式传输响应。

        def iterator() -> Iterator:
            yield first_response  # 返回第一个响应
            yield from iterator_or_completion  # 继续返回迭代器中的剩余响应

        send_chan, recv_chan = anyio.create_memory_object_stream(10)  # 创建内存对象流通道
        return EventSourceResponse(
            recv_chan,  # 使用接收通道
            data_sender_callable=partial(
                get_event_publisher,  # 调用获取事件发布者
                request=raw_request,  # 原始请求对象
                inner_send_chan=send_chan,  # 内部发送通道
                iterator=iterator(),  # 使用迭代器作为数据源
            ),
        )
    else:
        return iterator_or_completion  # 直接返回非迭代器对象作为响应

