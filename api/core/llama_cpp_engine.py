from typing import (
    Optional,
    List,
    Union,
    Dict,
    Iterator,
    Any,
)

from llama_cpp import Llama
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from api.adapter import get_prompt_adapter
from api.utils.compat import model_validate


class LlamaCppEngine:
    def __init__(
        self,
        model: Llama,
        model_name: str,
        prompt_name: Optional[str] = None,
    ):
        """
        Initializes a LlamaCppEngine instance.

        Args:
            model (Llama): The Llama model to be used by the engine.
            model_name (str): The name of the model.
            prompt_name (Optional[str], optional): The name of the prompt. Defaults to None.
        """
        self.model = model  # 初始化模型对象
        self.model_name = model_name.lower()  # 将模型名称转换为小写
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None  # 将提示名称转换为小写（如果存在）
        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)
        # 获取并设置适配器用于处理提示

    def apply_chat_template(
            self,
            messages: List[ChatCompletionMessageParam],
            functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Applies a chat template to the given list of messages.

        Args:
            messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional):
                The functions to be applied to the messages. Defaults to None.
            tools (Optional[List[Dict[str, Any]]], optional):
                The tools to be used for postprocessing the messages. Defaults to None.

        Returns:
            str: The chat template applied to the messages.
        """
        if self.prompt_adapter.function_call_available:  # 检查适配器是否支持函数调用
            messages = self.prompt_adapter.postprocess_messages(messages, functions, tools)
            # 如果支持，对消息进行后处理，应用函数和工具

        return self.prompt_adapter.apply_chat_template(messages)
        # 应用聊天模板到处理过的消息上，并返回结果字符串

    def create_completion(self, prompt, **kwargs) -> Union[Iterator[dict], Dict[str, Any]]:
        """
        Creates a completion using the specified prompt and additional keyword arguments.

        Args:
            prompt (str): The prompt for the completion.
            **kwargs: Additional keyword arguments to be passed to the model's create_completion method.

        Returns:
            Union[Iterator[dict], Dict[str, Any]]: The completion generated by the model.
        """
        return self.model.create_completion(prompt, **kwargs)
        # 调用模型的 create_completion 方法，传入指定的 prompt 和额外的关键字参数，并返回生成的完成结果

    def _create_chat_completion(self, prompt, **kwargs) -> ChatCompletion:
        """
        Creates a chat completion using the specified prompt and additional keyword arguments.

        Args:
            prompt (str): The prompt for the chat completion.
            **kwargs: Additional keyword arguments to be passed to the create_completion method.

        Returns:
            ChatCompletion: The chat completion generated by the model.
        """
        # 生成完成结果
        completion = self.create_completion(prompt, **kwargs)

        # 创建聊天完成消息
        message = ChatCompletionMessage(
            role="assistant",
            content=completion["choices"][0]["text"].strip(),  # 获取完成结果中的文本内容
        )

        # 创建选择项
        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop",
            logprobs=None,
        )

        # 验证完成用法
        usage = model_validate(CompletionUsage, completion["usage"])

        # 返回聊天完成对象
        return ChatCompletion(
            id="chat" + completion["id"],  # 构建完成的唯一标识符
            choices=[choice],
            created=completion["created"],  # 完成生成的时间戳
            model=completion["model"],  # 使用的模型名称
            object="chat.completion",  # 对象类型为聊天完成
            usage=usage,
        )

    def _create_chat_completion_stream(self, prompt, **kwargs) -> Iterator[ChatCompletionChunk]:
        """
        Generates a stream of chat completion chunks based on the given prompt.

        Args:
            prompt (str): The prompt for generating chat completion chunks.
            **kwargs: Additional keyword arguments for creating completions.

        Yields:
            ChatCompletionChunk: A chunk of chat completion generated from the prompt.
        """
        # 生成完成结果流
        completion = self.create_completion(prompt, **kwargs)

        for i, output in enumerate(completion):
            _id, _created, _model = output["id"], output["created"], output["model"]

            # 如果是第一个输出，创建空选择项作为初始块
            if i == 0:
                choice = ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=""),
                    finish_reason=None,
                    logprobs=None,
                )
                yield ChatCompletionChunk(
                    id=f"chat{_id}",
                    choices=[choice],
                    created=_created,
                    model=_model,
                    object="chat.completion.chunk",
                )

            # 获取当前输出的完成原因和文本内容，构建选择项
            if output["choices"][0]["finish_reason"] is None:
                delta = ChoiceDelta(content=output["choices"][0]["text"])  # 获取文本内容
            else:
                delta = ChoiceDelta()  # 创建空的完成块

            choice = ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=output["choices"][0]["finish_reason"],  # 获取完成原因
                logprobs=None,
            )
            yield ChatCompletionChunk(
                id=f"chat{_id}",
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
            )

    def create_chat_completion(self, prompt, **kwargs) -> Union[Iterator[ChatCompletionChunk], ChatCompletion]:

        return (
            self._create_chat_completion_stream(prompt, **kwargs)  # 如果参数中有 stream=True，返回流式聊天完成结果
            if kwargs.get("stream", False)
            else self._create_chat_completion(prompt, **kwargs)  # 否则返回单个聊天完成结果
        )

    @property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            The stop property of the prompt adapter, or None if it does not exist.
        """
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None  # 返回 prompt adapter 的 stop 属性，如果不存在则返回 None
