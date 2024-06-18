import time
import traceback
import uuid
from functools import partial
from typing import AsyncIterator

import anyio
import vllm
from fastapi import APIRouter, Depends, status
from fastapi import HTTPException, Request
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage
from sse_starlette import EventSourceResponse
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

<<<<<<< HEAD
from api.core.vllm_engine import VllmEngine
from api.models import LLM_ENGINE
from api.utils.compat import dictify, model_validate
from api.utils.protocol import Role, ChatCompletionCreateParams
from api.utils.request import (
    check_api_key,
    handle_request,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")  # 创建一个带有"/chat"前缀的APIRouter对象

vllm_version = vllm.__version__  # 获取vllm库的版本号

def get_engine():
    yield LLM_ENGINE  # 生成LLM_ENGINE对象的生成器

@chat_router.post(
    "/completions",
    dependencies=[Depends(check_api_key)],  # POST请求处理"/chat/completions"路径，依赖于check_api_key函数进行身份验证
    status_code=status.HTTP_200_OK,  # 设置状态码为200表示请求成功
)
async def create_chat_completion(
    request: ChatCompletionCreateParams,  # 请求参数类型为ChatCompletionCreateParams
    raw_request: Request,  # 原始HTTP请求对象
    engine: VllmEngine = Depends(get_engine),  # 引擎依赖于get_engine函数返回的VllmEngine对象
):
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")  # 如果请求中没有消息或最后一个消息的角色为ASSISTANT，则抛出HTTP异常

    request = await handle_request(request, engine.prompt_adapter.stop)  # 处理请求，包括设置停止条件
    request.max_tokens = request.max_tokens or 512  # 如果未设置最大令牌数，则默认为512

    params = dictify(request, exclude={"messages"})  # 将请求参数转换为字典，排除messages字段
    params.update(dict(prompt_or_messages=request.messages, echo=False))  # 更新参数字典，包括消息和echo字段
    logger.debug(f"==== request ====\n{params}")  # 记录调试日志，输出请求参数

    request_id: str = f"chatcmpl-{str(uuid.uuid4())}"  # 生成请求ID

    prompt = engine.apply_chat_template(
        request.messages,
        functions=request.functions,
        tools=request.tools,
    )  # 应用聊天模板生成提示

    if isinstance(prompt, list):
        prompt, token_ids = None, prompt  # 如果提示是列表，则置为None，token_ids为提示列表
    else:
        prompt, token_ids = prompt, None  # 否则，使用prompt作为提示，token_ids置为None

    token_ids = engine.convert_to_inputs(prompt, token_ids, max_tokens=request.max_tokens)  # 将提示和token_ids转换为输入
    result_generator = None  # 初始化结果生成器为None
=======
from api.common import dictify, model_validate
from api.engine.vllm_engine import VllmEngine
from api.models import LLM_ENGINE
from api.protocol import Role, ChatCompletionCreateParams
from api.utils import (
    check_api_key,
    check_completion_requests,
    get_event_publisher,
)

chat_router = APIRouter(prefix="/chat")
vllm_version = vllm.__version__


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
    engine: VllmEngine = Depends(get_engine),
):
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT.value:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await check_completion_requests(
        request,
        engine.template.stop,
        engine.template.stop_token_ids,
    )
    request.max_tokens = request.max_tokens or 512

    if request.best_of < request.n:
        request.best_of = request.n

    params = dictify(request, exclude={"messages"})
    params.update(dict(prompt_or_messages=request.messages, echo=False))
    logger.debug(f"==== request ====\n{params}")

    request_id: str = f"chatcmpl-{str(uuid.uuid4())}"
    token_ids = engine.template.convert_messages_to_ids(
        messages=request.messages,
        tools=request.tools,
        max_tokens=request.max_tokens,
    )

    result_generator = None
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    try:
        include = {
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
<<<<<<< HEAD
        }  # 包含在参数中的字段集合
        kwargs = dictify(request, include=include)  # 将请求参数转换为字典，包括指定字段
        sampling_params = SamplingParams(
            stop=request.stop or [],  # 停止条件
            stop_token_ids=request.stop_token_ids or [],  # 停止token的ID列表
            max_tokens=request.max_tokens,  # 最大令牌数
            **kwargs,  # 其他参数
        )
        lora_request = engine._maybe_get_lora(request.model)  # 获取LoRA请求

        try:
            from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor

            if vllm_version >= "0.4.2":  # 如果vllm版本大于或等于0.4.2
                decoding_config = await engine.model.get_decoding_config()  # 获取解码配置
=======
        }
        kwargs = dictify(request, include=include)
        sampling_params = SamplingParams(
            stop=request.stop or [],
            stop_token_ids=request.stop_token_ids or [],
            max_tokens=request.max_tokens,
            **kwargs,
        )

        # Todo: support for lora
        lora_request = None
        try:
            from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor

            if vllm_version >= "0.4.2":
                decoding_config = await engine.model.get_decoding_config()
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                guided_decode_logits_processor = (
                    await get_guided_decoding_logits_processor(
                        request.guided_decoding_backend or decoding_config.guided_decoding_backend,
                        request,
                        engine.tokenizer,
                    )
                )
            else:
                guided_decode_logits_processor = (
                    await get_guided_decoding_logits_processor(
                        request,
                        engine.tokenizer,
                    )
                )
            if guided_decode_logits_processor:
                sampling_params.logits_processors = sampling_params.logits_processors or []
                sampling_params.logits_processors.append(guided_decode_logits_processor)
        except ImportError:
            pass

<<<<<<< HEAD
        if vllm_version >= "0.4.3":  # 如果vllm版本大于或等于0.4.3
            result_generator = engine.model.generate(
                {
                    "prompt": prompt if isinstance(prompt, str) else None,
=======
        if vllm_version >= "0.4.3":
            result_generator = engine.model.generate(
                {
                    "prompt": None,
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                    "prompt_token_ids": token_ids,
                },
                sampling_params,
                request_id,
                lora_request,
            )
        else:
            result_generator = engine.model.generate(
<<<<<<< HEAD
                prompt if isinstance(prompt, str) else None,
=======
                None,
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                sampling_params,
                request_id,
                token_ids,
                lora_request,
            )

    except ValueError as e:
<<<<<<< HEAD
        traceback.print_exc()  # 打印异常的跟踪信息

    if request.stream:  # 如果请求需要流式处理
        iterator = create_chat_completion_stream(result_generator, request, request_id, engine)  # 创建结果生成器的流式处理迭代器
        send_chan, recv_chan = anyio.create_memory_object_stream(10)  # 创建内存对象流通道

=======
        traceback.print_exc()

    if request.stream:
        iterator = create_chat_completion_stream(result_generator, request, request_id, engine)
        send_chan, recv_chan = anyio.create_memory_object_stream(10)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator,
            ),
<<<<<<< HEAD
        )  # 返回EventSourceResponse对象，用于推送事件流给客户端

    else:
        # 非流式响应
        final_res: RequestOutput = None
        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():  # 如果客户端断开连接
                await engine.model.abort(request_id)  # 终止模型处理
                return  # 返回空值

            final_res = res  # 设置最终结果为当前结果

        assert final_res is not None  # 确保最终结果不为空
        choices = []
        for output in final_res.outputs:
            output.text = output.text.replace("�", "")  # 替换特殊字符

            finish_reason = output.finish_reason  # 获取结束原因
            function_call = None
            if request.functions or request.tools:  # 如果存在函数或工具调用
                try:
                    res, function_call = engine.prompt_adapter.parse_assistant_response(
                        output.text, request.functions, request.tools,
                    )  # 解析助手响应
                    output.text = res  # 设置输出文本为解析结果
                except Exception as e:
                    traceback.print_exc()  # 打印异常的跟踪信息
                    logger.warning("Failed to parse tool call")  # 记录警告日志，解析工具调用失败

            if isinstance(function_call, dict) and "arguments" in function_call:  # 如果存在参数的函数调用
                function_call = FunctionCall(**function_call)  # 创建FunctionCall对象
=======
        )
    else:
        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                await engine.model.abort(request_id)
                return
            final_res = res

        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            output.text = output.text.replace("�", "")

            finish_reason = output.finish_reason
            function_call = None
            if request.functions or request.tools:
                try:
                    res, function_call = engine.template.parse_assistant_response(
                        output.text, request.tools or request.functions,
                    )
                    output.text = res
                except Exception as e:
                    traceback.print_exc()
                    logger.warning("Failed to parse tool call")

            if isinstance(function_call, dict) and "arguments" in function_call:
                function_call = FunctionCall(**function_call)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    function_call=function_call
<<<<<<< HEAD
                )  # 创建助手消息对象，包括函数调用
                finish_reason = "function_call"  # 设置结束原因为函数调用
            elif isinstance(function_call, dict) and "function" in function_call:  # 如果存在函数的工具调用
                finish_reason = "tool_calls"  # 设置结束原因为工具调用
                tool_calls = [model_validate(ChatCompletionMessageToolCall, function_call)]  # 验证工具调用
=======
                )
                finish_reason = "function_call"
            elif isinstance(function_call, dict) and "function" in function_call:
                finish_reason = "tool_calls"
                tool_calls = [model_validate(ChatCompletionMessageToolCall, function_call)]
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    tool_calls=tool_calls,
<<<<<<< HEAD
                )  # 创建助手消息对象，包括工具调用
            else:
                message = ChatCompletionMessage(role="assistant", content=output.text.strip())  # 创建助手消息对象，仅包括文本内容
=======
                )
            else:
                message = ChatCompletionMessage(role="assistant", content=output.text.strip())
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d

            choices.append(
                Choice(
                    index=output.index,
                    message=message,
                    finish_reason=finish_reason,
                )
<<<<<<< HEAD
            )  # 添加选择项

        num_prompt_tokens = len(final_res.prompt_token_ids)  # 计算提示令牌数
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)  # 计算生成令牌数
=======
            )

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
<<<<<<< HEAD
        )  # 创建完成使用对象

        return ChatCompletion(
            id=request_id,
            choices=choices,
            created=int(time.time()),  # 创建时间戳
            model=request.model,
            object="chat.completion",
            usage=usage,
        )  # 返回聊天完成对象，包括请求ID、选择项、创建时间戳、模型、对象类型和使用情况信息


async def create_chat_completion_stream(
        generator: AsyncIterator,  # 异步生成器，产生RequestOutput对象
        request: ChatCompletionCreateParams,  # 聊天完成请求参数对象
        request_id: str,  # 请求ID
        engine: VllmEngine,  # VllmEngine对象，用于处理聊天完成请求
) -> AsyncIterator:
    for i in range(request.n):  # 遍历请求中指定的生成次数n
        # 创建第一个块（chunk），角色为助手，内容为空
        choice = ChunkChoice(
            index=i,  # 索引
            delta=ChoiceDelta(role="assistant", content=""),  # 选择增量对象，角色为助手，内容为空
            finish_reason=None,  # 结束原因为空
            logprobs=None,  # 日志概率为空
        )
        yield ChatCompletionChunk(
            id=request_id,  # 请求ID
            choices=[choice],  # 选择项列表
            created=int(time.time()),  # 创建时间戳
            model=request.model,  # 模型名称
            object="chat.completion.chunk",  # 对象类型为聊天完成块
        )

        previous_texts = [""] * request.n  # 存储先前文本内容的列表，长度为n，初始为空字符串
        previous_num_tokens = [0] * request.n  # 存储先前令牌数的列表，长度为n，初始为0
        async for res in generator:  # 异步迭代结果生成器
            res: RequestOutput  # 将res视为RequestOutput对象
            for output in res.outputs:  # 遍历每个输出对象
                i = output.index  # 获取输出对象的索引
                output.text = output.text.replace("�", "")  # 替换特殊字符

                delta_text = output.text[len(previous_texts[i]):]  # 计算增量文本，从先前文本内容到当前输出文本的差异部分
                previous_texts[i] = output.text  # 更新先前文本内容
                previous_num_tokens[i] = len(output.token_ids)  # 更新先前令牌数

                finish_reason = output.finish_reason  # 获取输出对象的结束原因
                delta = None  # 初始化增量对象为空

                if finish_reason is None:  # 如果结束原因为空
                    delta = ChoiceDelta(content=delta_text)  # 创建内容为增量文本的ChoiceDelta对象
                elif request.functions or request.tools:  # 如果存在函数或工具调用
                    call_info = None
                    try:
                        res, call_info = engine.prompt_adapter.parse_assistant_response(
                            output.text, request.functions, request.tools,
                        )  # 解析助手响应中的函数或工具调用信息
                    except Exception as e:
                        traceback.print_exc()  # 打印异常的跟踪信息
                        logger.warning("Failed to parse tool call")  # 记录警告日志，解析工具调用失败

                    if isinstance(call_info, dict) and "arguments" in call_info:  # 如果调用信息是字典且包含参数
                        finish_reason = "function_call"  # 设置结束原因为函数调用
                        function_call = ChoiceDeltaFunctionCall(**call_info)  # 创建函数调用的ChoiceDeltaFunctionCall对象
                        delta = ChoiceDelta(
                            role="assistant",  # 角色为助手
                            content=delta_text,  # 增量文本内容
                            function_call=function_call,  # 函数调用信息
                        )
                    elif isinstance(call_info, dict) and "function" in call_info:  # 如果调用信息是字典且包含函数
                        finish_reason = "tool_calls"  # 设置结束原因为工具调用
                        call_info["index"] = 0
                        tool_calls = [model_validate(ChoiceDeltaToolCall, call_info)]  # 验证工具调用信息
                        delta = ChoiceDelta(
                            role="assistant",  # 角色为助手
                            content=delta_text,  # 增量文本内容
                            tool_calls=tool_calls,  # 工具调用信息
                        )

                choice = ChunkChoice(
                    index=i,
                    delta=delta or ChoiceDelta(content=delta_text),  # 如果增量不为空则使用增量，否则使用增量文本内容
                    finish_reason=finish_reason,  # 结束原因
                    logprobs=None,  # 日志概率为空
                )
                yield ChatCompletionChunk(
                    id=request_id,  # 请求ID
                    choices=[choice],  # 选择项列表
                    created=int(time.time()),  # 创建时间戳
                    model=request.model,  # 模型名称
                    object="chat.completion.chunk",  # 对象类型为聊天完成块
                )  # 生成聊天完成块对象的生成器

=======
        )
        return ChatCompletion(
            id=request_id,
            choices=choices,
            created=int(time.time()),
            model=request.model,
            object="chat.completion",
            usage=usage,
        )


async def create_chat_completion_stream(
    generator: AsyncIterator,
    request: ChatCompletionCreateParams,
    request_id: str,
    engine: VllmEngine,
) -> AsyncIterator:
    for i in range(request.n):
        # First chunk with role
        choice = ChunkChoice(
            index=i,
            delta=ChoiceDelta(role="assistant", content=""),
            finish_reason=None,
            logprobs=None,
        )
        yield ChatCompletionChunk(
            id=request_id,
            choices=[choice],
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
        )

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                output.text = output.text.replace("�", "")

                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)

                finish_reason = output.finish_reason
                delta = None

                if finish_reason is None:
                    delta = ChoiceDelta(content=delta_text)
                elif request.functions or request.tools:
                    call_info = None
                    try:
                        res, call_info = engine.template.parse_assistant_response(
                            output.text, request.tools or request.functions,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        logger.warning("Failed to parse tool call")

                    if isinstance(call_info, dict) and "arguments" in call_info:
                        finish_reason = "function_call"
                        function_call = ChoiceDeltaFunctionCall(**call_info)
                        delta = ChoiceDelta(
                            role="assistant",
                            content=delta_text,
                            function_call=function_call
                        )
                    elif isinstance(call_info, dict) and "function" in call_info:
                        finish_reason = "tool_calls"
                        call_info["index"] = 0
                        tool_calls = [model_validate(ChoiceDeltaToolCall, call_info)]
                        delta = ChoiceDelta(
                            role="assistant",
                            content=delta_text,
                            tool_calls=tool_calls,
                        )
                
                choice = ChunkChoice(
                    index=i,
                    delta=delta or ChoiceDelta(content=delta_text),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
                yield ChatCompletionChunk(
                    id=request_id,
                    choices=[choice],
                    created=int(time.time()),
                    model=request.model,
                    object="chat.completion.chunk",
                )
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
