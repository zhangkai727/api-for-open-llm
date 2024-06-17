import json
from typing import (
    Optional,
    List,
    AsyncIterator,
)

from aiohttp import ClientSession
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
from text_generation import AsyncClient
from text_generation.errors import parse_error
from text_generation.types import Request, Parameters
from text_generation.types import Response, StreamResponse

from api.adapter import get_prompt_adapter
from api.utils.compat import dictify


class TGIEngine:
    def __init__(
        self,
        model: AsyncClient,
        model_name: str,
        prompt_name: Optional[str] = None,
    ):
        """
        Initializes the TGIEngine object.

        Args:
            model (AsyncClient): The asynchronous client for the model.
            model_name (str): The name of the model.
            prompt_name (Optional[str], optional): The name of the prompt. Defaults to None.
        """
        self.model = model  # 设置模型对象
        self.model_name = model_name.lower()  # 将模型名称转换为小写
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None  # 如果有提示名称，则转换为小写
        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)
        # 获取相应的提示适配器根据模型名称和提示名称


    def apply_chat_template(
        self, messages: List[ChatCompletionMessageParam],
    ) -> str:
        """
        Applies a chat template to the given messages and returns the processed output.

        Args:
            messages: A list of ChatCompletionMessageParam objects representing the chat messages.

        Returns:
            str: The processed output as a string.
        """
        return self.prompt_adapter.apply_chat_template(messages)

    async def generate(
        self,
        prompt: str,
        do_sample: bool = True,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        decoder_input_details: bool = True,
        top_n_tokens: Optional[int] = None,
    ) -> Response:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of the highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step

        Returns:
            Response: generated response
        """
        parameters = Parameters(
            best_of=best_of,  # 设置生成过程中的 best_of 参数。
            details=True,  # 表示请求详细的解码器输入详情。
            decoder_input_details=decoder_input_details,  # 传递 decoder_input_details 参数。
            do_sample=do_sample,  # 设置是否激活 logits 抽样。
            max_new_tokens=max_new_tokens,  # 定义要生成的最大标记数。
            repetition_penalty=repetition_penalty,  # 设置重复惩罚参数。
            return_full_text=return_full_text,  # 决定是否将输入提示前置到生成的文本中。
            seed=seed,  # 设置随机抽样种子。
            stop=stop_sequences if stop_sequences is not None else [],  # 如果提供了停止序列，则指定停止序列。
            temperature=temperature,  # 设置调节 logits 分布的温度参数。
            top_k=top_k,  # 设置 top-k 过滤的最高概率词汇标记数。
            top_p=top_p,  # 设置 nucleus 抽样的 top-p 参数。
            truncate=truncate,  # 设置截断输入标记的大小。
            typical_p=typical_p,  # 指定用于解码质量的 typical_p 参数。
            watermark=watermark,  # 表示是否应用水印。
            top_n_tokens=top_n_tokens,  # 指定每步返回的最有可能的标记数。
        )

        # 构造一个 Request 对象，包括输入的提示文本、禁用流模式以及定义的参数。
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        # 使用异步会话向模型的 /generate 端点发送 POST 请求。
        async with ClientSession(
                headers=self.model.headers, cookies=self.model.cookies, timeout=self.model.timeout
        ) as session:
            async with session.post(f"{self.model.base_url}/generate", json=dictify(request)) as resp:
                payload = await resp.json()

                # 如果响应状态码不是 200，则抛出一个解析错误，包括状态码和响应数据。
                if resp.status != 200:
                    raise parse_error(resp.status, payload)

                # 返回从响应数据实例化的 Response 对象。
                return Response(**payload)

    async def generate_stream(
            self,
            prompt: str,  # 输入的提示文本
            do_sample: bool = False,  # 是否激活 logits 抽样，默认为 False
            max_new_tokens: int = 20,  # 最大生成标记数，默认为 20
            best_of: Optional[int] = 1,  # 生成多个序列并返回具有最高标记 logprobs 的序列数，默认为 1
            repetition_penalty: Optional[float] = None,  # 重复惩罚参数
            return_full_text: bool = False,  # 是否将输入提示前置到生成的文本中，默认为 False
            seed: Optional[int] = None,  # 随机抽样种子
            stop_sequences: Optional[List[str]] = None,  # 停止生成标记序列的标记列表
            temperature: Optional[float] = None,  # 调节 logits 分布的温度参数
            top_k: Optional[int] = None,  # top-k 过滤的最高概率词汇标记数
            top_p: Optional[float] = None,  # nucleus 抽样的 top-p 参数
            truncate: Optional[int] = None,  # 截断输入标记的大小
            typical_p: Optional[float] = None,  # 典型解码质量的 typical_p 参数
            watermark: bool = False,  # 是否应用水印
            top_n_tokens: Optional[int] = None,  # 每步返回的最有可能的标记数
    ) -> AsyncIterator[StreamResponse]:

        """
        Given a prompt, generate the following stream of tokens asynchronously

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of the highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step

        Returns:
            AsyncIterator: stream of generated tokens
        """
        # Validate parameters
        # 验证参数

        parameters = Parameters(
            best_of=best_of,  # 生成多个序列并返回具有最高标记 logprobs 的序列数
            details=True,  # 返回详细信息
            do_sample=do_sample,  # 是否激活 logits 抽样
            max_new_tokens=max_new_tokens,  # 最大生成标记数
            repetition_penalty=repetition_penalty,  # 重复惩罚参数
            return_full_text=return_full_text,  # 是否返回完整文本
            seed=seed,  # 随机种子
            stop=stop_sequences if stop_sequences is not None else [],  # 停止生成标记序列的标记列表
            temperature=temperature,  # 温度参数，用于调节 logits 分布
            top_k=top_k,  # top-k 过滤的最高概率词汇标记数
            top_p=top_p,  # nucleus 抽样的 top-p 参数
            truncate=truncate,  # 截断输入标记的大小
            typical_p=typical_p,  # 典型解码质量的 typical_p 参数
            watermark=watermark,  # 是否应用水印
            top_n_tokens=top_n_tokens,  # 每步返回的最有可能的标记数
        )

        request = Request(inputs=prompt, parameters=parameters)
        # 创建一个请求对象，将输入提示和参数传递给请求对象

        async with ClientSession(
                headers=self.model.headers, cookies=self.model.cookies, timeout=self.model.timeout
        ) as session:
            async with session.post(f"{self.model.base_url}/generate_stream", json=dictify(request)) as resp:
                if resp.status != 200:
                    raise parse_error(resp.status, await resp.json())

                # Parse ServerSentEvents
                async for byte_payload in resp.content:
                    # 跳过空行
                    if byte_payload == b"\n":
                        continue

                    payload = byte_payload.decode("utf-8")

                    # Event data
                    if payload.startswith("data:"):
                        # 解码有效负载
                        json_payload = json.loads(payload.lstrip("data:").rstrip("\n"))
                        # 解析有效负载
                        try:
                            response = StreamResponse(**json_payload)
                        except ValidationError:
                            # 如果解析有效负载失败，则为错误有效负载
                            raise parse_error(resp.status, json_payload)
                        yield response

    @property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            The stop property of the prompt adapter, or None if it does not exist.
        """
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
