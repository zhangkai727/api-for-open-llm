import asyncio
import time
import traceback
import uuid
from functools import partial
from typing import AsyncIterator, Tuple

import anyio
import vllm
from fastapi import APIRouter, Depends
from fastapi import Request
from loguru import logger
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice, Logprobs
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

from api.core.vllm_engine import VllmEngine
from api.models import LLM_ENGINE
from api.utils.compat import dictify
from api.utils.protocol import CompletionCreateParams
from api.utils.request import (
    handle_request,
    get_event_publisher,
    check_api_key,
)

completion_router = APIRouter()  # 创建API路由器对象

vllm_version = vllm.__version__  # 获取VLLM库的版本号


def get_engine():
    yield LLM_ENGINE  # 返回LLM_ENGINE的生成器


def parse_prompt_format(prompt) -> Tuple[bool, list]:
    # 解析提示格式函数，返回布尔值和列表
    # openai支持的格式包括字符串、字符串数组、令牌数组或令牌数组的数组
    prompt_is_tokens = False  # 默认提示不是令牌数组
    prompts = [prompt]  # 默认为单个提示字符串的列表

    if isinstance(prompt, list):  # 如果提示是列表
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")  # 提示至少要提供一个元素
        elif isinstance(prompt[0], str):
            prompt_is_tokens = False  # 提示是字符串数组
            prompts = prompt  # 使用字符串数组作为提示
        elif isinstance(prompt[0], int):
            prompt_is_tokens = True  # 提示是令牌数组
            prompts = [prompt]  # 使用令牌数组作为提示
        elif isinstance(prompt[0], list) and isinstance(prompt[0][0], int):
            prompt_is_tokens = True  # 提示是令牌数组的数组
            prompts = prompt  # 使用令牌数组的数组作为提示
        else:
            raise ValueError(
                "prompt must be a string, array of strings, array of tokens, or array of token arrays"
            )  # 提示必须是字符串、字符串数组、令牌数组或令牌数组的数组中的一种格式

    return prompt_is_tokens, prompts  # 返回提示是否为令牌数组以及解析后的提示列表


def merge_async_iterators(*iterators):
    """将多个异步迭代器合并为单个迭代器。

    这个方法处理一些迭代器先完成的情况。
    当产生结果时，它返回一个元组(i, item)，其中i是产生item的迭代器的索引。
    """
    queue = asyncio.Queue()  # 创建异步队列用于存储迭代器的输出结果

    finished = [False] * len(iterators)  # 创建一个列表，记录每个迭代器是否完成的状态，默认为False

    async def producer(i, iterator):
        try:
            async for item in iterator:  # 异步迭代每个迭代器的输出结果
                await queue.put((i, item))  # 将结果以元组(i, item)的形式放入异步队列中
        except Exception as e:
            await queue.put(e)  # 如果迭代器出现异常，将异常信息放入异步队列中
        finished[i] = True  # 标记当前迭代器已完成

    _tasks = [
        asyncio.create_task(producer(i, iterator))  # 创建异步任务，每个任务处理一个迭代器
        for i, iterator in enumerate(iterators)  # 枚举所有迭代器，为每个迭代器创建一个任务
    ]

    async def consumer():
        while not all(finished) or not queue.empty():  # 当所有迭代器都完成并且队列不为空时循环
            item = await queue.get()  # 从异步队列中获取结果
            if isinstance(item, Exception):
                raise item  # 如果结果是异常，则抛出异常
            yield item  # 返回正常结果

        await asyncio.gather(*_tasks)  # 等待所有任务完成

    return consumer()  # 返回合并后的异步迭代器函数


@completion_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_completion(
    request: CompletionCreateParams,
    raw_request: Request,
    engine: VllmEngine = Depends(get_engine),
):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.
    """
    request.max_tokens = request.max_tokens or 128  # 如果未指定最大令牌数，则默认为128
    request = await handle_request(request, engine.prompt_adapter.stop, chat=False)  # 处理请求

    if isinstance(request.prompt, list):
        request.prompt = request.prompt[0]  # 如果提示是列表，则取第一个元素作为提示字符串

    params = dictify(request, exclude={"prompt"})  # 生成请求参数字典，排除提示参数
    params.update(dict(prompt_or_messages=request.prompt))  # 更新参数字典，包含提示或消息
    logger.debug(f"==== request ====\n{params}")  # 记录调试信息

    request_id: str = f"cmpl-{str(uuid.uuid4())}"  # 生成请求ID
    generators = []  # 生成器列表，用于存储生成器对象
    num_prompts = 1  # 提示数量，默认为1

    try:
        include = {  # 包含在采样参数中的字段
            "n",
            "presence_penalty",
            "frequency_penalty",
            "temperature",
            "top_p",
            "repetition_penalty",
            "min_p",
            "best_of",
            "ignore_eos",
            "use_beam_search",
            "skip_special_tokens",
            "spaces_between_special_tokens",
        }
        kwargs = dictify(request, include=include)  # 生成包含指定字段的请求参数字典
        sampling_params = SamplingParams(  # 创建采样参数对象
            stop=request.stop or [],  # 停止令牌列表，默认为空列表
            stop_token_ids=request.stop_token_ids or [],  # 停止令牌ID列表，默认为空列表
            max_tokens=request.max_tokens,  # 最大令牌数
            **kwargs,  # 包含其他请求参数
        )
        lora_request = engine._maybe_get_lora(request.model)  # 获取LORA请求对象

        try:
            from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor

            if vllm_version >= "0.4.2":
                decoding_config = await engine.model.get_decoding_config()  # 获取解码配置
                guided_decode_logits_processor = (
                    await get_guided_decoding_logits_processor(  # 获取引导解码逻辑处理器
                        request.guided_decoding_backend or decoding_config.guided_decoding_backend,
                        request,
                        engine.tokenizer,
                    )
                )
            else:
                guided_decode_logits_processor = (
                    await get_guided_decoding_logits_processor(  # 获取引导解码逻辑处理器
                        request,
                        engine.tokenizer,
                    )
                )
            if guided_decode_logits_processor:
                sampling_params.logits_processors = sampling_params.logits_processors or []
                sampling_params.logits_processors.append(guided_decode_logits_processor)  # 添加引导解码处理器
        except ImportError:
            pass

        prompt_is_tokens, prompts = parse_prompt_format(request.prompt)  # 解析提示格式
        num_prompts = len(prompts)  # 计算提示数量

        for i, prompt in enumerate(prompts):
            if prompt_is_tokens:
                input_ids = engine.convert_to_inputs(token_ids=prompt, max_tokens=request.max_tokens)
            else:
                input_ids = engine.convert_to_inputs(prompt=prompt, max_tokens=request.max_tokens)

            if vllm_version >= "0.4.3":
                generator = engine.model.generate(  # 生成结果生成器对象
                    {
                        "prompt": prompt,
                        "prompt_token_ids": input_ids,
                    },
                    sampling_params,
                    request_id,
                    lora_request,
                )
            else:
                generator = engine.model.generate(  # 生成结果生成器对象
                    prompt,
                    sampling_params,
                    f"{request_id}-{i}",
                    prompt_token_ids=input_ids,
                    lora_request=lora_request
                )

            generators.append(generator)  # 将结果生成器添加到列表中
    except ValueError as e:
        traceback.print_exc()  # 打印异常信息

    result_generator: AsyncIterator[Tuple[int, RequestOutput]] = merge_async_iterators(*generators)  # 合并生成器

    if request.stream:  # 如果是流式响应
        iterator = create_completion_stream(  # 创建流式迭代器
            engine, result_generator, request, request_id, num_prompts
        )
        send_chan, recv_chan = anyio.create_memory_object_stream(10)  # 创建内存对象流
        return EventSourceResponse(  # 返回事件源响应
            recv_chan,
            data_sender_callable=partial(  # 部分应用发送数据的可调用函数
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator,
            ),
        )
    else:
        # 非流式响应
        final_res_batch = [None] * num_prompts  # 初始化最终结果批次列表
        async for i, res in result_generator:
            if await raw_request.is_disconnected():
                # 如果客户端断开连接，则中止请求
                await engine.model.abort(f"{request_id}-{i}")
            final_res_batch[i] = res  # 存储每个提示的最终结果

        choices = []  # 初始化选择列表
        num_prompt_tokens = 0  # 初始化提示令牌数量
        num_generated_tokens = 0  # 初始化生成的令牌数量

        for final_res in final_res_batch:
            final_res: RequestOutput
            prompt_token_ids = final_res.prompt_token_ids  # 获取提示的令牌ID
            prompt_logprobs = final_res.prompt_logprobs  # 获取提示的对数概率
            prompt_text = final_res.prompt  # 获取提示文本

            for output in final_res.outputs:
                if request.echo and request.max_tokens == 0:
                    token_ids = prompt_token_ids
                    top_logprobs = prompt_logprobs
                    output_text = prompt_text
                elif request.echo and request.max_tokens > 0:
                    token_ids = prompt_token_ids + output.token_ids
                    top_logprobs = (prompt_logprobs + output.logprobs if request.logprobs else None)
                    output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    top_logprobs = output.logprobs
                    output_text = output.text

                if request.logprobs is not None:
                    logprobs = engine.create_logprobs(  # 创建对数概率
                        token_ids=token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=request.logprobs,
                    )
                else:
                    logprobs = None

                choice = CompletionChoice(  # 创建完成选项对象
                    index=len(choices),
                    text=output_text,
                    finish_reason=output.finish_reason,
                    logprobs=logprobs,
                )
                choices.append(choice)  # 添加选择对象到列表中

                num_prompt_tokens += len(prompt_token_ids)  # 更新提示令牌数量
                num_generated_tokens += sum(len(output.token_ids) for output in final_res.outputs)  # 更新生成的令牌数量

        usage = CompletionUsage(  # 创建完成使用情况对象
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return Completion(  # 返回完成对象
            id=request_id,
            choices=choices,
            created=int(time.time()),
            model=request.model,
            object="text_completion",
            usage=usage,
        )


async def create_completion_stream(
    engine: VllmEngine,  # 引擎对象，用于处理生成的结果
    generator: AsyncIterator,  # 异步生成器，生成模型输出结果
    request: CompletionCreateParams,  # 完成请求的参数对象
    request_id: str,  # 请求的唯一标识符
    num_prompts: int,  # 提示数量
) -> AsyncIterator:
    # 初始化列表，用于存储先前的文本、令牌数和回显状态
    previous_texts = [""] * request.n * num_prompts  # 存储先前文本的列表，每个提示对应一个
    previous_num_tokens = [0] * request.n * num_prompts  # 存储先前令牌数量的列表，每个提示对应一个
    has_echoed = [False] * request.n * num_prompts  # 存储是否已回显的列表，每个提示对应一个

    try:
        # 异步迭代生成器，获取结果和索引
        async for prompt_idx, res in generator:
            res: RequestOutput  # 将结果标记为RequestOutput类型
            for output in res.outputs:
                i = output.index + prompt_idx * request.n  # 计算当前输出在结果数组中的索引位置

                output.text = output.text.replace("�", "")  # 替换输出文本中的非法字符

                if request.echo and request.max_tokens == 0:
                    # 仅返回提示文本，不包含生成的文本
                    delta_text = res.prompt
                    delta_token_ids = res.prompt_token_ids
                    top_logprobs = res.prompt_logprobs
                    has_echoed[i] = True  # 标记已回显
                elif request.echo and request.max_tokens > 0 and not has_echoed[i]:
                    # 返回提示文本及生成的第一个令牌
                    delta_text = res.prompt + output.text
                    delta_token_ids = res.prompt_token_ids + output.token_ids
                    top_logprobs = res.prompt_logprobs + (output.logprobs or [])
                    has_echoed[i] = True  # 标记已回显
                else:
                    # 仅返回生成的文本增量
                    delta_text = output.text[len(previous_texts[i]):]
                    delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                    top_logprobs = (
                        output.logprobs[previous_num_tokens[i]:] if output.logprobs else None
                    )

                if request.logprobs is not None:
                    # 如果请求中包含logprobs，则生成logprobs对象
                    assert top_logprobs is not None, (
                        "top_logprobs must be provided when logprobs "
                        "is requested"
                    )
                    logprobs = engine.create_logprobs(
                        token_ids=delta_token_ids,
                        top_logprobs=top_logprobs,
                        num_output_top_logprobs=request.logprobs,
                        initial_text_offset=len(previous_texts[i]),  # 初始文本偏移量
                    )
                else:
                    logprobs = None

                # 更新先前文本和令牌数量
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                # 创建完成选项对象
                choice = CompletionChoice(
                    index=i,  # 选项索引
                    text=delta_text,  # 选项文本
                    finish_reason="stop",  # 完成原因设置为"stop"
                    logprobs=logprobs,  # logprobs对象
                )

                # 生成完成对象
                yield Completion(
                    id=request_id,  # 请求的唯一标识符
                    choices=[choice],  # 完成选项列表
                    created=int(time.time()),  # 创建时间戳
                    model=request.model,  # 模型名称
                    object="text_completion",  # 对象类型为"text_completion"
                )

                # 如果存在完成原因，则生成额外的完成对象
                if output.finish_reason is not None:
                    if request.logprobs is not None:
                        logprobs = Logprobs(
                            text_offset=[], token_logprobs=[], tokens=[], top_logprobs=[]
                        )
                    else:
                        logprobs = None

                    choice = CompletionChoice(
                        index=i,  # 选项索引
                        text=delta_text,  # 选项文本
                        finish_reason="stop",  # 完成原因设置为"stop"
                        logprobs=logprobs,  # logprobs对象
                    )

                    yield Completion(
                        id=request_id,  # 请求的唯一标识符
                        choices=[choice],  # 完成选项列表
                        created=int(time.time()),  # 创建时间戳
                        model=request.model,  # 模型名称
                        object="text_completion",  # 对象类型为"text_completion"
                    )

    except:
        traceback.print_exc()  # 捕获并打印异常堆栈信息

