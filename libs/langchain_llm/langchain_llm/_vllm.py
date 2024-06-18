from __future__ import annotations

import time
import traceback
import uuid
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from langchain_community.llms.vllm import VLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.pydantic_v1 import root_validator
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion import Completion
from openai.types.completion_choice import (
    CompletionChoice,
)
from openai.types.completion_usage import CompletionUsage

from ._compat import model_parse
from .adapters.template import (
    get_prompt_adapter,
    BaseTemplate,
)
from .generation import (
    build_qwen_chat_input,
)


class XVLLM(VLLM):
    """vllm model."""

    model_name: str
    """The name of a HuggingFace Transformers model."""

    def call_as_openai(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Completion:
        """Run the LLM on the given prompt and input."""

        from vllm import SamplingParams

<<<<<<< HEAD
        # 构建采样参数
        params = {**self._default_params, **kwargs, "stop": stop}
        sampling_params = SamplingParams(**params)
        # 调用模型生成文本
        outputs = self.client.generate([prompt], sampling_params)[0]

        choices = []
        # 解析模型输出并生成完成选项
=======
        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}
        sampling_params = SamplingParams(**params)
        # call the model
        outputs = self.client.generate([prompt], sampling_params)[0]

        choices = []
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        for output in outputs.outputs:
            text = output.text
            choices.append(
                CompletionChoice(
                    index=0,
                    text=text,
                    finish_reason="stop",
                    logprobs=None,
                )
            )

<<<<<<< HEAD
        # 统计生成的token数目
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        num_prompt_tokens = len(outputs.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in outputs.outputs)
        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

<<<<<<< HEAD
        # 返回完成对象
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        return Completion(
            id=f"cmpl-{str(uuid.uuid4())}",
            choices=choices,
            created=int(time.time()),
            model=self.model_name,
            object="text_completion",
            usage=usage,
        )


class ChatVLLM(BaseChatModel):
    """
    Wrapper for using VLLM as ChatModels.
    """

    llm: XVLLM

    chat_template: Optional[str] = None
    """Chat template for generating completions."""

    max_window_size: Optional[int] = 6144
    """The maximum window size"""

    construct_prompt: bool = True

    prompt_adapter: Optional[BaseTemplate] = None

    tokenizer: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

<<<<<<< HEAD
        # 检查是否需要构建提示或输入。
=======
        # Check whether to need to construct prompts or inputs. """
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        model_name = values["llm"].model_name
        values["chat_template"] = values["chat_template"].lower() if values["chat_template"] is not None else None
        if not values["prompt_adapter"]:
            try:
<<<<<<< HEAD
                # 尝试获取适配器以处理模型名称和聊天模板
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                values["prompt_adapter"] = get_prompt_adapter(model_name, values["chat_template"])
            except KeyError:
                values["chat_template"] = None

<<<<<<< HEAD
        # 获取模型的分词器
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        values["tokenizer"] = values["llm"].client.get_tokenizer()

        return values

    def _get_parameters(
<<<<<<< HEAD
            self,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        准备模型推理所需的参数，包括停止序列。

        Args:
            stop (Optional[List[str]]): 停止序列的列表。

        Returns:
            Dict[str, Any]: 包含默认参数和用户提供的kwargs的合并参数字典。
=======
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Performs sanity check, preparing parameters.

        Args:
            stop (Optional[List[str]]): List of stop sequences.

        Returns:
            Dictionary containing the combined parameters.
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        """

        params = self.llm._default_params

<<<<<<< HEAD
        # 从prompt_adapter中获取停止序列及其对应的token IDs
=======
        # then sets it as configured, or default to an empty list:
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        _stop, _stop_token_ids = [], []
        if isinstance(self.prompt_adapter.stop, dict):
            _stop_token_ids = self.prompt_adapter.stop.get("token_ids", [])
            _stop = self.prompt_adapter.stop.get("strings", [])

<<<<<<< HEAD
        # 确保stop是一个列表，如果为None则默认为空列表
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        stop = stop or []
        if isinstance(stop, str):
            stop = [stop]

<<<<<<< HEAD
        # 合并并去重来自prompt_adapter和用户输入的停止序列
        params["stop"] = list(set(_stop + stop))
        params["stop_token_ids"] = list(set(_stop_token_ids))

        # 将params与用户提供的额外kwargs合并
=======
        params["stop"] = list(set(_stop + stop))
        params["stop_token_ids"] = list(set(_stop_token_ids))

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        params = {**params, **kwargs}

        return params

    def _generate(
<<<<<<< HEAD
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        from vllm import SamplingParams

        # 将消息转换为聊天提示所需的格式
        llm_input = self._to_chat_prompt(messages)
        params = self._get_parameters(stop, **kwargs)

        # 构建用于推理的采样参数
        sampling_params = SamplingParams(**params)

        # 调用语言模型根据提示和采样参数生成文本
=======
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from vllm import SamplingParams

        llm_input = self._to_chat_prompt(messages)
        params = self._get_parameters(stop, **kwargs)

        # build sampling parameters
        sampling_params = SamplingParams(**params)
        # call the model

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        if isinstance(llm_input, str):
            prompts, prompt_token_ids = [llm_input], None
        else:
            prompts, prompt_token_ids = None, [llm_input]

        outputs = self.llm.client.generate(prompts, sampling_params, prompt_token_ids)

        generations = []
<<<<<<< HEAD
        # 将模型输出处理为ChatGeneration对象
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        for output in outputs:
            text = output.outputs[0].text
            generations.append(ChatGeneration(message=AIMessage(content=text)))

        return ChatResult(generations=generations)

    def _to_chat_prompt(self, messages: List[BaseMessage]) -> Union[List[int], Dict[str, Any]]:
<<<<<<< HEAD
        """将消息列表转换为语言模型预期的格式。"""
        if not messages:
            raise ValueError("必须提供至少一个HumanMessage")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("最后一个消息必须是HumanMessage")

        # 将消息格式转换为ChatML格式
        messages_dicts = [self._to_chatml_format(m) for m in messages]

        # 应用聊天模板到转换后的消息格式
        return self._apply_chat_template(messages_dicts)

    def _apply_chat_template(
            self,
            messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
            functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, List[int]]:
        """
        应用聊天模板生成模型输入。

        Args:
            messages (Union[List[ChatCompletionMessageParam], Dict[str, Any]]): 聊天完成消息参数的列表或字典。
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): 应用到消息的函数。默认为None。
            tools (Optional[List[Dict[str, Any]]], optional): 应用到消息的工具。默认为None。

        Returns:
            Union[str, List[int]]: 生成的模型输入。
        """
        if self.prompt_adapter.function_call_available:
            # 如果可以调用函数，使用prompt_adapter对消息进行后处理
=======
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("at least one HumanMessage must be provided")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("last message must be a HumanMessage")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self._apply_chat_template(messages_dicts)

    def _apply_chat_template(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, List[int]]:
        """
        Apply chat template to generate model inputs.

        Args:
            messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.
            tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.

        Returns:
            Union[str, List[int]]: The generated inputs.
        """
        if self.prompt_adapter.function_call_available:
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            messages = self.prompt_adapter.postprocess_messages(
                messages, functions, tools,
            )
            if functions or tools:
                logger.debug(f"==== Messages with tools ====\n{messages}")

        if "chatglm3" in self.llm.model_name:
<<<<<<< HEAD
            # 如果模型名称包含"chatglm3"，使用tokenizer构建聊天输入
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            query, role = messages[-1]["content"], messages[-1]["role"]
            return self.tokenizer.build_chat_input(
                query, history=messages[:-1], role=role
            )["input_ids"][0].tolist()
        elif "qwen" in self.llm.model_name:
<<<<<<< HEAD
            # 如果模型名称包含"qwen"，使用build_qwen_chat_input构建聊天输入
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            return build_qwen_chat_input(
                self.tokenizer,
                messages,
                self.max_window_size,
                functions,
                tools,
            )
        else:
<<<<<<< HEAD
            # 否则，根据条件应用聊天模板到消息
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            if getattr(self.tokenizer, "chat_template", None) and not self.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self.prompt_adapter.apply_chat_template(messages)
            return prompt

<<<<<<< HEAD
    def call_as_openai(
            self,
            messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> ChatCompletion:
        # 调用模型并返回输出结果。
=======
    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @property
    def _llm_type(self) -> str:
        return "vllm-chat-wrapper"

    def call_as_openai(
        self,
        messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Call the model and return the output.
        """
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        from vllm import SamplingParams

        llm_input = self._apply_chat_template(
            messages,
            functions=kwargs.get("functions"),
            tools=kwargs.get("tools"),
        )
        params = self._get_parameters(stop, **kwargs)

<<<<<<< HEAD
        # 构建采样参数
        sampling_params = SamplingParams(**params)

        # 调用模型
=======
        # build sampling parameters
        sampling_params = SamplingParams(**params)
        # call the model

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        if isinstance(llm_input, str):
            prompts, prompt_token_ids = [llm_input], None
        else:
            prompts, prompt_token_ids = None, [llm_input]

        outputs = self.llm.client.generate(prompts, sampling_params, prompt_token_ids)[0]

        choices = []
        for output in outputs.outputs:
            function_call, finish_reason = None, "stop"
            if params.get("functions") or params.get("tools"):
                try:
                    res, function_call = self.prompt_adapter.parse_assistant_response(
                        output.text, params.get("functions"), params.get("tools"),
                    )
                    output.text = res
                except Exception as e:
                    traceback.print_exc()
                    logger.warning("Failed to parse tool call")

            if isinstance(function_call, dict) and "arguments" in function_call:
                finish_reason = "function_call"
                function_call = FunctionCall(**function_call)
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    function_call=function_call,
                )
            elif isinstance(function_call, dict) and "function" in function_call:
                finish_reason = "tool_calls"
                tool_calls = [model_parse(ChatCompletionMessageToolCall, function_call)]
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text,
                    tool_calls=tool_calls,
                )
            else:
                message = ChatCompletionMessage(
                    role="assistant",
                    content=output.text.strip(),
                )

            choices.append(
                Choice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            )

        num_prompt_tokens = len(outputs.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in outputs.outputs)
        usage = CompletionUsage(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return ChatCompletion(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            choices=choices,
            created=int(time.time()),
            model=self.llm.model_name,
            object="chat.completion",
            usage=usage,
        )
<<<<<<< HEAD

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        # 将 LangChain 消息转换为 ChatML 格式。
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @property
    def _llm_type(self) -> str:
        # 返回 VLLM 包装的类型标识字符串。
        return "vllm-chat-wrapper"

    def _apply_chat_template(
            self,
            messages: Union[List[ChatCompletionMessageParam], Dict[str, Any]],
            functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, List[int]]:
        """
        Apply chat template to generate model inputs.

        Args:
            messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.
            tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.

        Returns:
            Union[str, List[int]]: The generated inputs.
        """
        if self.prompt_adapter.function_call_available:
            messages = self.prompt_adapter.postprocess_messages(
                messages, functions, tools,
            )
            if functions or tools:
                logger.debug(f"==== Messages with tools ====\n{messages}")

        if "chatglm3" in self.llm.model_name:
            query, role = messages[-1]["content"], messages[-1]["role"]
            return self.tokenizer.build_chat_input(
                query, history=messages[:-1], role=role
            )["input_ids"][0].tolist()
        elif "qwen" in self.llm.model_name:
            return build_qwen_chat_input(
                self.tokenizer,
                messages,
                self.max_window_size,
                functions,
                tools,
            )
        else:
            if getattr(self.tokenizer, "chat_template", None) and not self.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self.prompt_adapter.apply_chat_template(messages)
            return prompt

=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
