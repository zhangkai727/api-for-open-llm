import asyncio
import time
from dataclasses import dataclass
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Union,
)

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from openai.types.completion_choice import Logprobs
from openai.types.model import Model
from pydantic import BaseModel
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.transformers_utils.tokenizer import get_tokenizer

from api.adapter import get_prompt_adapter
from api.generation import build_qwen_chat_input


@dataclass
class LoRA:
    name: str
    local_path: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[Model] = []


class VllmEngine:
    def __init__(
        self,
        model: AsyncLLMEngine,
        model_name: str,
        prompt_name: Optional[str] = None,
        lora_modules: Optional[List[LoRA]] = None,
    ):
        # 根据提供的模块初始化 LoRA 请求列表
        if lora_modules is None:
            self.lora_requests = []
        else:
            try:
                from vllm.lora.request import LoRARequest
                self.lora_requests = [
                    LoRARequest(
                        lora_name=lora.name,
                        lora_int_id=i,
                        lora_local_path=lora.local_path,
                    ) for i, lora in enumerate(lora_modules, start=1)
                ]
            except ImportError:
                self.lora_requests = []

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        # 如果有运行中的事件循环，安排执行 _post_init 方法
        if event_loop is not None and event_loop.is_running():
            # 如果当前实例由 Ray Serve 管理，已经有运行中的事件循环
            event_loop.create_task(self._post_init())
        else:
            # 当单独使用单个 vLLM 实例而不使用 engine_use_ray 时
            asyncio.run(self._post_init())

    async def _post_init(self):
        # 获取引擎模型的配置信息
        engine_model_config = await self.model.get_model_config()
        # 设置最大模型长度
        self.max_model_len = engine_model_config.max_model_len

        # 使用独立的分词器将 token ID 映射为字符串
        self.tokenizer = get_tokenizer(
            engine_model_config.tokenizer,
            tokenizer_mode=engine_model_config.tokenizer_mode,
            trust_remote_code=engine_model_config.trust_remote_code,
        )

    async def show_available_models(self) -> ModelList:
        """展示可用的模型。当前我们只有一个模型。"""
        # 创建主模型卡片列表
        model_cards = [
            Model(
                id=self.model_name,
                object="model",
                created=int(time.time()),
                owned_by="vllm"
            )
        ]

        # 创建 LoRA 模块的卡片列表（如果有的话）
        lora_cards = [
            Model(
                id=lora.lora_name,
                object="model",
                created=int(time.time()),
                owned_by="vllm"
            )
            for lora in self.lora_requests
        ]

        # 合并主模型卡片和 LoRA 模块卡片
        model_cards.extend(lora_cards)

        # 返回模型列表对象
        return ModelList(data=model_cards)

    def create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: Optional[List[Optional[Any]]] = None,
        num_output_top_logprobs: Optional[int] = None,
        initial_text_offset: int = 0,
    ):
        """Create OpenAI-style logprobs."""
        logprobs = Logprobs()
        logprobs.tokens = []  # 存储生成的 token
        logprobs.token_logprobs = []  # 存储每个 token的logprobs
        logprobs.text_offset = []  # 存储每个 token 的文本偏移量

        last_token_len = 0  # 上一个 token 的长度
        if num_output_top_logprobs:
            logprobs.top_logprobs = []  # 存储每个 token 的 top logprobs

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]  # 获取当前 token 的 top logprobs
            if step_top_logprobs is None:
                token = self.tokenizer.decode(token_id)  # 解码 token
                logprobs.tokens.append(token)  # 存储解码后的 token
                logprobs.token_logprobs.append(None)  # 将 logprobs 设置为 None
                logprobs.top_logprobs.append(None)  # 将 top logprobs 设置为 None
            else:
                token_logprob = step_top_logprobs[token_id].logprob  # 获取当前 token 的 logprob
                token = step_top_logprobs[token_id].decoded_token  # 获取当前 token 的解码结果
                logprobs.tokens.append(token)  # 存储解码后的 token
                logprobs.token_logprobs.append(token_logprob)  # 存储当前 token 的 logprob

                if num_output_top_logprobs:
                    # 如果需要输出 top logprobs，则构建字典存储每个 top logprob 的解码 token 和 logprob
                    logprobs.top_logprobs.append(
                        {
                            p.decoded_token: p.logprob
                            for i, p in step_top_logprobs.items()
                        }
                        if step_top_logprobs else None
                    )

            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)  # 第一个 token 的文本偏移量
            else:
                # 计算当前 token 的文本偏移量，累加上一个 token 的长度
                logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
            last_token_len = len(token)  # 更新上一个 token 的长度

        return logprobs

    def _maybe_get_lora(self, model_name):
        for lora in self.lora_requests:  # 遍历所有的 LoRA 请求对象列表
            if model_name == lora.lora_name:  # 如果找到名称匹配的 LoRA 请求对象
                logger.info(f"Lora request: {model_name}")  # 记录日志信息
                return lora  # 返回匹配的 LoRA 请求对象
        return None  # 如果未找到匹配的 LoRA 请求对象，则返回 None

    def apply_chat_template(
        self,
        messages: List[ChatCompletionMessageParam],
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, List[int]]:
        """
        Applies a chat template to the given messages and returns the processed output.

        Args:
            messages: A list of ChatCompletionMessageParam objects representing the chat messages.
            functions: A dictionary or list of dictionaries representing the functions to be applied (optional).
            tools: A list of dictionaries representing the tools to be used (optional).

        Returns:
            Union[str, List[int]]: The processed output as a string or a list of integers.
        """
        if self.prompt_adapter.function_call_available:  # 检查是否支持函数调用
            messages = self.prompt_adapter.postprocess_messages(  # 后处理消息
                messages, functions, tools,
            )
            if functions or tools:  # 如果指定了函数或工具
                logger.debug(f"==== Messages with tools ====\n{messages}")  # 记录调试信息

        if "chatglm3" in self.model_name:  # 如果模型名称包含 "chatglm3"
            query, role = messages[-1]["content"], messages[-1]["role"]  # 获取查询和角色
            return self.tokenizer.build_chat_input(  # 构建聊天输入
                query, history=messages[:-1], role=role
            )["input_ids"][0].tolist()  # 返回构建的聊天输入的 token IDs 列表
        elif "chatglm4" in self.model_name:  # 如果模型名称包含 "chatglm4"
            return self.tokenizer.apply_chat_template(  # 应用聊天模板
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        elif self.model_name.startswith("qwen") and ("qwen1.5" not in self.model_name) and (
                "qwen2" not in self.model_name):
            return build_qwen_chat_input(  # 构建 QWEN 的聊天输入
                self.tokenizer,
                messages,
                functions=functions,
                tools=tools,
            )
        else:
            if getattr(self.tokenizer, "chat_template", None) and not self.prompt_name:  # 检查是否有聊天模板且未指定 prompt 名称
                logger.debug("Using tokenizer's chat template")  # 记录调试信息
                prompt = self.tokenizer.apply_chat_template(  # 应用 tokenizer 的聊天模板
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self.prompt_adapter.apply_chat_template(messages)  # 否则，应用 prompt 适配器的聊天模板
            return prompt  # 返回处理后的输出

    def convert_to_inputs(
        self,
        prompt: Optional[str] = None,
        token_ids: Optional[List[int]] = None,
        max_tokens: Optional[int] = 256,
    ) -> List[int]:
        input_ids = token_ids or self.tokenizer(prompt).input_ids  # 如果没有指定 token_ids，则从 prompt 转换
        input_len = len(input_ids)  # 获取输入 token IDs 的长度
        min_max_tokens = 256  # 最小的最大 token 数量
        if input_len > self.max_model_len - min_max_tokens:  # 如果输入长度超过最大模型长度减去最小最大 token 数量
            max_input_tokens = self.max_model_len - min_max_tokens  # 设置最大输入 token 数量
        else:
            max_input_tokens = max(self.max_model_len - max_tokens, input_len)  # 否则设置为最大模型长度减去指定的 max_tokens 或输入长度
        return input_ids[-max_input_tokens:]  # 返回截取后的输入 token IDs 列表


@property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            The stop property of the prompt adapter, or None if it does not exist.
        """
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
