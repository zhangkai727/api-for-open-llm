import time
import uuid
from functools import partial
from typing import (
    Dict,
    Any,
    AsyncIterator,
)

import anyio
from fastapi import APIRouter, Depends, status
from fastapi import Request
from loguru import logger
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from text_generation.types import Response, StreamResponse

from api.core.tgi import TGIEngine
from api.models import LLM_ENGINE
from api.utils.compat import dictify
from api.utils.protocol import CompletionCreateParams
from api.utils.request import (
    handle_request,
    get_event_publisher,
    check_api_key
)

completion_router = APIRouter()


def get_engine():
    yield LLM_ENGINE  # 返回语言模型引擎对象 LLM_ENGINE


@completion_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],  # API密钥验证依赖
    status_code=status.HTTP_200_OK,  # 响应状态码为200 OK
)
async def create_completion(
    request: CompletionCreateParams,  # 请求参数为CompletionCreateParams对象
    raw_request: Request,  # 原始请求对象
    engine: TGIEngine = Depends(get_engine),  # 使用get_engine函数获取语言模型引擎
):
    """ Completion API similar to OpenAI's API. """
    request.max_tokens = request.max_tokens or 128  # 设置最大生成token数量，默认为128
    request = await handle_request(request, engine.prompt_adapter.stop, chat=False)  # 处理请求，停止适配器，chat参数为False

    if isinstance(request.prompt, list):
        request.prompt = request.prompt[0]  # 如果提示文本是列表，则取第一个元素作为提示文本

    request_id: str = f"cmpl-{str(uuid.uuid4())}"  # 生成请求ID
    include = {"temperature", "best_of", "repetition_penalty", "typical_p", "watermark"}  # 包含在参数中的属性
    params = dictify(request, include=include)  # 将请求转换为字典形式，包含指定的属性
    params.update(
        dict(
            prompt=request.prompt,  # 设置提示文本
            do_sample=request.temperature > 1e-5,  # 根据温度是否大于1e-5设置是否进行采样
            max_new_tokens=request.max_tokens,  # 设置最大生成新token数量
            stop_sequences=request.stop,  # 设置停止序列
            top_p=request.top_p if request.top_p < 1.0 else 0.99,  # 设置top-p值
            return_full_text=request.echo,  # 设置是否返回完整文本
        )
    )
    logger.debug(f"==== request ====\n{params}")  # 记录调试信息

    if request.stream:  # 如果是流式处理
        generator = engine.generate_stream(**params)  # 生成流对象
        iterator = create_completion_stream(generator, params, request_id)  # 创建完成流迭代器
        send_chan, recv_chan = anyio.create_memory_object_stream(10)  # 创建内存对象流
        return EventSourceResponse(  # 返回事件源响应
            recv_chan,  # 接收通道
            data_sender_callable=partial(  # 部分发送数据的调用
                get_event_publisher,  # 获取事件发布者
                request=raw_request,  # 原始请求
                inner_send_chan=send_chan,  # 内部发送通道
                iterator=iterator,  # 迭代器
            ),
        )

    # 非流式响应
    response: Response = await engine.generate(**params)  # 生成响应

    finish_reason = response.details.finish_reason.value  # 完成原因
    finish_reason = "length" if finish_reason == "length" else "stop"  # 如果完成原因为length则设置为stop
    choice = CompletionChoice(  # 完成选项
        index=0,  # 索引为0
        text=response.generated_text,  # 生成的文本内容
        finish_reason=finish_reason,  # 完成原因
        logprobs=None,  # 概率日志为空
    )

    num_prompt_tokens = len(response.details.prefill)  # 提示token数量
    num_generated_tokens = response.details.generated_tokens  # 生成的token数量
    usage = CompletionUsage(  # 完成使用情况
        prompt_tokens=num_prompt_tokens,  # 提示token数量
        completion_tokens=num_generated_tokens,  # 完成token数量
        total_tokens=num_prompt_tokens + num_generated_tokens,  # 总token数量
    )

    return Completion(  # 返回完成对象
        id=request_id,  # 请求ID
        choices=[choice],  # 完成选项列表
        created=int(time.time()),  # 创建时间戳
        model=params.get("model", "llm"),  # 模型名称，默认为llm
        object="text_completion",  # 对象类型为text_completion
        usage=usage,  # 使用情况
    )



async def create_completion_stream(
    generator: AsyncIterator[StreamResponse],  # 异步迭代器，生成流响应对象
    params: Dict[str, Any],  # 参数字典，包含请求的参数信息
    request_id: str,  # 请求ID，用于唯一标识每个完成流
) -> AsyncIterator[Completion]:
    async for output in generator:
        output: StreamResponse
        if output.token.special:
            continue  # 如果输出的token为特殊token，则继续下一个循环

        choice = CompletionChoice(
            index=0,  # 索引为0
            text=output.token.text,  # 生成的文本内容
            finish_reason="stop",  # 完成原因为stop
            logprobs=None,  # 概率日志为空
        )
        yield Completion(
            id=request_id,  # 请求ID
            choices=[choice],  # 完成选项列表
            created=int(time.time()),  # 创建时间戳
            model=params.get("model", "llm"),  # 模型名称，默认为llm
            object="text_completion",  # 对象类型为text_completion
        )

