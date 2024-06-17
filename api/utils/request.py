import json
from threading import Lock
from typing import (
    Optional,
    Union,
    Iterator,
    Dict,
    Any,
    AsyncIterator,
)

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from api.config import SETTINGS
from api.utils.compat import jsonify, dictify
from api.utils.constants import ErrorCode
from api.utils.protocol import (
    ChatCompletionCreateParams,
    CompletionCreateParams,
    ErrorResponse,
)

llama_outer_lock = Lock()
llama_inner_lock = Lock()


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):
    if not SETTINGS.api_keys:  # 如果没有设置 API keys
        return None  # 允许所有请求通过
    if auth is None or (token := auth.credentials) not in SETTINGS.api_keys:  # 如果没有 Authorization 或者 token 不在 API keys 中
        raise HTTPException(  # 抛出 401 错误
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
    return token  # 返回 token


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(dictify(ErrorResponse(message=message, code=code)), status_code=500)  # 创建一个包含错误信息的 JSONResponse 对象


async def handle_request(
    request: Union[CompletionCreateParams, ChatCompletionCreateParams],
    stop: Dict[str, Any] = None,
    chat: bool = True,
) -> Union[CompletionCreateParams, ChatCompletionCreateParams, JSONResponse]:
    error_check_ret = check_requests(request)  # 检查请求的参数是否有效
    if error_check_ret is not None:  # 如果有错误返回
        return error_check_ret  # 返回错误响应

    _stop, _stop_token_ids = [], []  # 初始化停止条件
    if stop is not None:  # 如果有传入停止条件
        _stop_token_ids = stop.get("token_ids", [])  # 获取停止 token IDs
        _stop = stop.get("strings", [])  # 获取停止字符串列表

    request.stop = request.stop or []  # 如果没有停止条件则初始化为空列表
    if isinstance(request.stop, str):  # 如果停止条件是字符串则转换为列表
        request.stop = [request.stop]

    if chat and ("qwen" in SETTINGS.model_name.lower() and (request.functions is not None or request.tools is not None)):  # 如果是聊天请求并且设置了特定的 model_name
        request.stop.append("Observation:")  # 添加停止条件

    request.stop = list(set(_stop + request.stop))  # 合并停止条件并去重
    request.stop_token_ids = request.stop_token_ids or []  # 初始化停止 token IDs
    request.stop_token_ids = list(set(_stop_token_ids + request.stop_token_ids))  # 合并停止 token IDs 并去重

    request.top_p = max(request.top_p, 1e-5)  # 确保 top_p 至少为 1e-5
    if request.temperature <= 1e-5:  # 如果 temperature 小于等于 1e-5
        request.top_p = 1.0  # 设置 top_p 为 1.0

    return request  # 返回处理后的请求参数


def check_requests(request: Union[CompletionCreateParams, ChatCompletionCreateParams]) -> Optional[JSONResponse]:
    if request.max_tokens is not None and request.max_tokens <= 0:  # 检查 max_tokens 参数是否有效
        return create_error_response(  # 返回 max_tokens 参数错误的 JSONResponse
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:  # 检查 n 参数是否有效
        return create_error_response(  # 返回 n 参数错误的 JSONResponse
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:  # 检查 temperature 参数是否有效
        return create_error_response(  # 返回 temperature 参数错误的 JSONResponse
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:  # 检查 temperature 参数是否有效
        return create_error_response(  # 返回 temperature 参数错误的 JSONResponse
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:  # 检查 top_p 参数是否有效
        return create_error_response(  # 返回 top_p 参数错误的 JSONResponse
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:  # 检查 top_p 参数是否有效
        return create_error_response(  # 返回 top_p 参数错误的 JSONResponse
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is None or isinstance(request.stop, (str, list)):  # 检查 stop 参数是否有效
        return None  # stop 参数有效，返回 None
    else:
        return create_error_response(  # 返回 stop 参数错误的 JSONResponse
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )



async def get_event_publisher(
    request: Request,  # HTTP 请求对象
    inner_send_chan: MemoryObjectSendStream,  # 内部内存流发送通道
    iterator: Union[Iterator, AsyncIterator],  # 迭代器，可以是同步或异步迭代器
):
    async with inner_send_chan:  # 使用内部发送通道进行异步上下文管理
        try:
            if SETTINGS.engine not in ["vllm", "tgi"]:  # 如果引擎不是 "vllm" 或 "tgi"
                async for chunk in iterate_in_threadpool(iterator):  # 在线程池中迭代处理数据块
                    if isinstance(chunk, BaseModel):  # 如果数据块是 Pydantic BaseModel 类型
                        chunk = jsonify(chunk)  # 转换为 JSON 字符串
                    elif isinstance(chunk, dict):  # 如果数据块是字典类型
                        chunk = json.dumps(chunk, ensure_ascii=False)  # 转换为 JSON 字符串

                    await inner_send_chan.send(dict(data=chunk))  # 发送数据块到内部发送通道

                    if await request.is_disconnected():  # 如果客户端断开连接
                        raise anyio.get_cancelled_exc_class()()  # 抛出取消异常

                    if SETTINGS.interrupt_requests and llama_outer_lock.locked():  # 如果中断请求被设置并且锁定状态
                        await inner_send_chan.send(dict(data="[DONE]"))  # 发送完成标志到内部发送通道
                        raise anyio.get_cancelled_exc_class()()  # 抛出取消异常

            else:  # 如果引擎是 "vllm" 或 "tgi"
                async for chunk in iterator:  # 迭代处理数据块
                    chunk = jsonify(chunk)  # 转换为 JSON 字符串
                    await inner_send_chan.send(dict(data=chunk))  # 发送数据块到内部发送通道
                    if await request.is_disconnected():  # 如果客户端断开连接
                        raise anyio.get_cancelled_exc_class()()  # 抛出取消异常

            await inner_send_chan.send(dict(data="[DONE]"))  # 发送完成标志到内部发送通道

        except anyio.get_cancelled_exc_class() as e:  # 捕获取消异常
            logger.info("disconnected")  # 记录日志，客户端断开连接
            with anyio.move_on_after(1, shield=True):  # 在1秒后取消操作
                logger.info(f"Disconnected from client (via refresh/close) {request.client}")  # 记录日志，客户端通过刷新/关闭断开连接
                raise e  # 重新抛出取消异常

