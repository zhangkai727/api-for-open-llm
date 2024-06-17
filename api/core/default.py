import traceback
from abc import ABC
from typing import (
    Optional,
    List,
    Union,
    Tuple,
    Dict,
    Iterator,
    Any,
)

import torch
from fastapi.responses import JSONResponse
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice, Logprobs
from openai.types.completion_usage import CompletionUsage
from transformers import PreTrainedModel, PreTrainedTokenizer

from api.adapter import get_prompt_adapter
from api.generation import (
    build_baichuan_chat_input,
    check_is_baichuan,
    generate_stream_chatglm,
    check_is_chatglm,
    generate_stream_chatglm_v3,
    build_qwen_chat_input,
    check_is_qwen,
    generate_stream_v2,
    build_xverse_chat_input,
    check_is_xverse,
)
from api.generation.utils import get_context_length
from api.utils.compat import model_validate
from api.utils.constants import ErrorCode
from api.utils.request import create_error_response

server_error_msg = (  # 定义服务器错误消息
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"  # 服务器高流量导致的网络错误消息
)



class DefaultEngine(ABC):  # 定义一个继承自 ABC 的抽象基类 DefaultEngine
    """ 基于原生 transformers 实现的模型引擎 """

    def __init__(  # 定义初始化方法
        self,
        model: PreTrainedModel,  # 模型对象
        tokenizer: PreTrainedTokenizer,  # 分词器对象
        model_name: str,  # 模型名称
        context_len: Optional[int] = None,  # 上下文长度，默认为 None
        prompt_name: Optional[str] = None,  # 提示名称，默认为 None
    ) -> None:
        """
        Initialize the Default class.

        Args:
            model (PreTrainedModel): The pre-trained model.  # 参数说明：预训练模型
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.  # 参数说明：模型的分词器
            model_name (str): The name of the model.  # 参数说明：模型名称
            context_len (Optional[int], optional): The length of the context. Defaults to None.  # 参数说明：上下文长度，可选，默认为 None
            prompt_name (Optional[str], optional): The name of the prompt. Defaults to None.  # 参数说明：提示名称，可选，默认为 None
        """
        self.model = model  # 初始化模型
        self.tokenizer = tokenizer  # 初始化分词器
        self.device = model.device  # 设备类型与模型一致

        self.model_name = model_name.lower()  # 模型名称小写化
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None  # 提示名称小写化（如果存在）
        self.context_len = context_len  # 保存上下文长度

        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)  # 获取提示适配器

        self._prepare_for_generate()  # 调用内部方法进行生成前的准备工作
        self._patch_tokenizer()  # 调用内部方法对分词器进行补丁处理（可能是自定义设置）

    def _prepare_for_generate(self) -> None:
        """
        Prepare the object for text generation.

        1. Sets the appropriate generate stream function based on the model name and type.
        2. Updates the context length if necessary.
        3. Checks and constructs the prompt.
        4. Sets the context length if it is not already set.
        """
        self.generate_stream_func = generate_stream_v2  # 默认设置生成流函数为 generate_stream_v2

        if "chatglm3" in self.model_name:  # 如果模型名称包含 "chatglm3"
            self.generate_stream_func = generate_stream_chatglm_v3  # 设置生成流函数为 generate_stream_chatglm_v3
        elif "chatglm4" in self.model_name:  # 如果模型名称包含 "chatglm4"
            self.generate_stream_func = generate_stream_v2  # 保持生成流函数为 generate_stream_v2
        elif check_is_chatglm(self.model):  # 如果模型是 chatglm 系列
            self.generate_stream_func = generate_stream_chatglm  # 设置生成流函数为 generate_stream_chatglm
        elif check_is_qwen(self.model):  # 如果模型是 qwen 系列
            self.context_len = 8192 if self.context_len is None else self.context_len  # 如果未设置上下文长度，设置为 8192

        self._check_construct_prompt()  # 调用内部方法检查和构建提示

        if self.context_len is None:  # 如果上下文长度仍未设置
            self.context_len = get_context_length(self.model.config)  # 从模型配置中获取并设置上下文长度

    def _check_construct_prompt(self) -> None:
        """ Check whether to need to construct prompts or inputs. """
        self.construct_prompt = self.prompt_name is not None  # 检查是否需要构建提示，如果 prompt_name 不为 None，则需要构建提示

        if "chatglm4" in self.model_name:  # 如果模型名称包含 "chatglm4"
            self.construct_prompt = False  # 不需要构建提示
            logger.info("Using ChatGLM4 Model for Chat!")  # 记录日志信息：使用 ChatGLM4 模型进行对话
        elif "chatglm3" in self.model_name:  # 如果模型名称包含 "chatglm3"
            logger.info("Using ChatGLM3 Model for Chat!")  # 记录日志信息：使用 ChatGLM3 模型进行对话
        elif check_is_baichuan(self.model):  # 如果模型是 Baichuan 系列
            logger.info("Using Baichuan Model for Chat!")  # 记录日志信息：使用 Baichuan 模型进行对话
        elif check_is_qwen(self.model):  # 如果模型是 Qwen 系列
            logger.info("Using Qwen Model for Chat!")  # 记录日志信息：使用 Qwen 模型进行对话
        elif check_is_xverse(self.model):  # 如果模型是 Xverse 系列
            logger.info("Using Xverse Model for Chat!")  # 记录日志信息：使用 Xverse 模型进行对话
        else:
            self.construct_prompt = True  # 否则需要构建提示

    def _patch_tokenizer(self) -> None:
        """ 
        Fix the tokenizer by adding the end-of-sequence (eos) token 
        and the padding (pad) token if they are missing.
        """
        from api.adapter.patcher import patch_tokenizer  # 从 api.adapter.patcher 导入 patch_tokenizer 函数

        patch_tokenizer(self.tokenizer)  # 调用 patch_tokenizer 函数修复分词器，添加缺失的 eos 和 pad token

    def convert_to_inputs(
            self,
            prompt_or_messages: Union[List[ChatCompletionMessageParam], str],  # 提示或消息，类型可以是字符串或消息参数列表
            infilling: Optional[bool] = False,  # 是否执行填充，默认为 False
            suffix_first: Optional[bool] = False,  # 是否先附加后缀，默认为 False
            **kwargs,  # 额外的关键字参数
    ) -> Tuple[Union[List[int], Dict[str, Any]], Union[List[ChatCompletionMessageParam], str]]:
        """
        Convert the prompt or messages into input format for the model.

        Args:
            prompt_or_messages: The prompt or messages to be converted.  # 参数说明：要转换的提示或消息
            infilling: Whether to perform infilling.  # 参数说明：是否执行填充
            suffix_first: Whether to append the suffix first.  # 参数说明：是否先附加后缀
            **kwargs: Additional keyword arguments.  # 参数说明：额外的关键字参数

        Returns:
            Tuple containing the converted inputs and the prompt or messages.  # 返回值说明：包含转换后的输入和提示或消息的元组
        """
        # for completion
        if isinstance(prompt_or_messages, str):  # 如果 prompt_or_messages 是字符串
            if infilling:  # 如果执行填充
                inputs = self.tokenizer(  # 使用分词器将字符串转换为输入 ID
                    prompt_or_messages, suffix_first=suffix_first,
                ).input_ids
            elif check_is_qwen(self.model):  # 如果模型是 Qwen 系列
                inputs = self.tokenizer(  # 使用分词器将字符串转换为输入 ID，并允许所有特殊字符
                    prompt_or_messages, allowed_special="all", disallowed_special=()
                ).input_ids
            elif check_is_chatglm(self.model):  # 如果模型是 ChatGLM 系列
                inputs = self.tokenizer([prompt_or_messages], return_tensors="pt")  # 使用分词器将字符串转换为 PyTorch 张量格式的输入
            else:
                inputs = self.tokenizer(prompt_or_messages).input_ids  # 否则，使用分词器将字符串转换为输入 ID

            if isinstance(inputs, list):  # 如果 inputs 是列表
                max_src_len = self.context_len - kwargs.get("max_tokens", 256) - 1  # 计算最大源长度，保留一些空间给生成的 tokens
                inputs = inputs[-max_src_len:]  # 截取输入列表的最后 max_src_len 个元素

        else:  # 如果 prompt_or_messages 不是字符串（是消息参数列表）
            inputs, prompt_or_messages = self.apply_chat_template(prompt_or_messages, **kwargs)  # 应用聊天模板，将消息参数列表转换为输入

        return inputs, prompt_or_messages  # 返回转换后的输入和提示或消息

    def apply_chat_template(
            self,
            messages: List[ChatCompletionMessageParam],  # 聊天完成消息参数的列表
            max_new_tokens: Optional[int] = 256,  # 最大生成的新 token 数量，默认为 256
            functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,  # 应用于消息的函数，默认为 None
            tools: Optional[List[Dict[str, Any]]] = None,  # 应用于消息的工具，默认为 None
            **kwargs,  # 额外的关键字参数
    ) -> Tuple[Union[List[int], Dict[str, Any]], Optional[str]]:
        """
        Apply chat template to generate model inputs and prompt.

        Args:
            messages (List[ChatCompletionMessageParam]): List of chat completion message parameters.  # 参数说明：聊天完成消息参数的列表
            max_new_tokens (Optional[int], optional): Maximum number of new tokens to generate. Defaults to 256.  # 参数说明：最大生成的新 token 数量，默认为 256
            functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): Functions to apply to the messages. Defaults to None.  # 参数说明：应用于消息的函数，默认为 None
            tools (Optional[List[Dict[str, Any]]], optional): Tools to apply to the messages. Defaults to None.  # 参数说明：应用于消息的工具，默认为 None
            **kwargs: Additional keyword arguments.  # 参数说明：额外的关键字参数

        Returns:
            Tuple[Union[List[int], Dict[str, Any]], Union[str, None]]: Tuple containing the generated inputs and prompt.  # 返回值说明：包含生成的输入和提示的元组
        """
        if self.prompt_adapter.function_call_available:  # 如果提示适配器支持函数调用
            messages = self.prompt_adapter.postprocess_messages(  # 后处理消息，应用函数和工具
                messages, functions, tools=tools,
            )
            if functions or tools:  # 如果有函数或工具
                logger.debug(f"==== Messages with tools ====\n{messages}")  # 记录带工具的消息日志

        if self.construct_prompt:  # 如果需要构建提示
            if getattr(self.tokenizer, "chat_template", None) and not self.prompt_name:  # 如果分词器有聊天模板且没有指定提示名称
                prompt = self.tokenizer.apply_chat_template(  # 使用分词器的聊天模板生成提示
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self.prompt_adapter.apply_chat_template(messages)  # 使用提示适配器应用聊天模板生成提示

            if check_is_qwen(self.model):  # 如果模型是 Qwen 系列
                inputs = self.tokenizer(prompt, allowed_special="all",
                                        disallowed_special=()).input_ids  # 使用分词器将提示转换为输入 ID，允许所有特殊字符
            elif check_is_chatglm(self.model):  # 如果模型是 ChatGLM 系列
                inputs = self.tokenizer([prompt], return_tensors="pt")  # 使用分词器将提示转换为 PyTorch 张量格式的输入
            else:
                inputs = self.tokenizer(prompt).input_ids  # 否则，使用分词器将提示转换为输入 ID

            if isinstance(inputs, list):  # 如果 inputs 是列表
                max_src_len = self.context_len - max_new_tokens - 1  # 计算最大源长度，保留一些空间给生成的 tokens
                inputs = inputs[-max_src_len:]  # 截取输入列表的最后 max_src_len 个元素
            return inputs, prompt  # 返回生成的输入和提示
        else:
            inputs = self.build_chat_inputs(  # 如果不需要构建提示，构建聊天输入
                messages, max_new_tokens, functions, tools, **kwargs
            )
        return inputs, None  # 返回生成的输入和 None（因为没有生成提示）

    def build_chat_inputs(
            self,
            messages: List[ChatCompletionMessageParam],  # 聊天完成消息参数的列表
            max_new_tokens: Optional[int] = 256,  # 最大生成的新 token 数量，默认为 256
            functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,  # 应用于消息的函数，默认为 None
            tools: Optional[List[Dict[str, Any]]] = None,  # 应用于消息的工具，默认为 None
            **kwargs: Any,  # 额外的关键字参数
    ) -> List[int]:
        if "chatglm3" in self.model_name:  # 如果模型名称包含 "chatglm3"
            query, role = messages[-1]["content"], messages[-1]["role"]  # 获取最后一个消息的内容和角色
            inputs = self.tokenizer.build_chat_input(query, history=messages[:-1], role=role)  # 使用分词器构建聊天输入，包含历史消息和角色
        elif "chatglm4" in self.model_name:  # 如果模型名称包含 "chatglm4"
            inputs = self.tokenizer.apply_chat_template(  # 使用分词器应用聊天模板
                messages,
                add_generation_prompt=True,  # 添加生成提示
                tokenize=True,  # 进行分词
            )[0]  # 获取生成的输入
        elif check_is_baichuan(self.model):  # 如果模型是 Baichuan 系列
            inputs = build_baichuan_chat_input(  # 构建 Baichuan 聊天输入
                self.tokenizer, messages, self.context_len, max_new_tokens
            )
        elif check_is_qwen(self.model):  # 如果模型是 Qwen 系列
            inputs = build_qwen_chat_input(  # 构建 Qwen 聊天输入
                self.tokenizer, messages, functions=functions, tools=tools,
            )
        elif check_is_xverse(self.model):  # 如果模型是 Xverse 系列
            inputs = build_xverse_chat_input(  # 构建 Xverse 聊天输入
                self.tokenizer, messages, self.context_len, max_new_tokens
            )
        else:
            raise NotImplementedError  # 如果模型不属于上述任何系列，则抛出 NotImplementedError 异常
        return inputs  # 返回生成的输入

    def _generate(self, params: Dict[str, Any]) -> Iterator[dict]:
        """
        Generates text based on the given parameters.

        Args:
            params (Dict[str, Any]): A dictionary containing the parameters for text generation.  # 参数说明：包含文本生成参数的字典

        Yields:
            Iterator: A dictionary containing the generated text and error code.  # 返回值说明：包含生成文本和错误代码的字典迭代器
        """
        prompt_or_messages = params.get("prompt_or_messages")  # 获取参数中的 prompt_or_messages
        inputs, prompt = self.convert_to_inputs(  # 调用 convert_to_inputs 方法将 prompt_or_messages 转换为模型输入
            prompt_or_messages,
            infilling=params.get("infilling", False),  # 获取 infilling 参数，默认为 False
            suffix_first=params.get("suffix_first", False),  # 获取 suffix_first 参数，默认为 False
            max_new_tokens=params.get("max_tokens", 256),  # 获取 max_tokens 参数，默认为 256
            functions=params.get("functions"),  # 获取 functions 参数
            tools=params.get("tools"),  # 获取 tools 参数
        )
        params.update(dict(inputs=inputs, prompt=prompt))  # 更新 params 字典，添加 inputs 和 prompt

        try:
            for output in self.generate_stream_func(self.model, self.tokenizer,
                                                    params):  # 调用 generate_stream_func 方法生成文本
                output["error_code"] = 0  # 将 error_code 设置为 0，表示没有错误
                yield output  # 生成输出

        except torch.cuda.OutOfMemoryError as e:  # 捕获 CUDA 内存不足错误
            yield {
                "text": f"{server_error_msg}\n\n({e})",  # 返回错误信息
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,  # 将 error_code 设置为 CUDA 内存不足错误代码
            }

        except (ValueError, RuntimeError) as e:  # 捕获 ValueError 和 RuntimeError 错误
            traceback.print_exc()  # 打印错误堆栈信息
            yield {
                "text": f"{server_error_msg}\n\n({e})",  # 返回错误信息
                "error_code": ErrorCode.INTERNAL_ERROR,  # 将 error_code 设置为内部错误代码
            }

    def _create_completion_stream(self, params: Dict[str, Any]) -> Iterator[Completion]:
        """
        Generates a stream of completions based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for generating completions.  # 参数说明：生成完成的参数字典

        Yields:
            Iterator: A stream of completion objects.  # 返回值说明：完成对象的流迭代器
        """
        for output in self._generate(params):  # 遍历生成文本的迭代器
            if output["error_code"] != 0:  # 如果输出的错误代码不为 0
                yield output  # 返回错误输出
                return  # 终止生成

            logprobs = None
            if params.get("logprobs") and output["logprobs"]:  # 如果参数中有 logprobs 并且输出中有 logprobs
                logprobs = model_validate(Logprobs, output["logprobs"])  # 验证 logprobs

            choice = CompletionChoice(  # 创建完成选择对象
                index=0,
                text=output["delta"],  # 完成的文本内容
                finish_reason="stop",  # 结束原因设为 "stop"
                logprobs=logprobs,  # logprobs
            )
            yield Completion(  # 返回完成对象
                id=output["id"],  # 输出的 ID
                choices=[choice],  # 完成选择对象列表
                created=output["created"],  # 创建时间
                model=output["model"],  # 模型名称
                object="text_completion",  # 对象类型
            )

    def _create_completion(self, params: Dict[str, Any]) -> Union[Completion, JSONResponse]:
        """
        Creates a completion based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for creating the completion.  # 参数说明：用于创建完成的参数字典

        Returns:
            Union[Completion, JSONResponse]: The generated completion object or a JSON response in case of error.  # 返回值说明：生成的完成对象或在错误情况下的 JSON 响应
        """
        last_output = None
        for output in self._generate(params):  # 遍历生成文本的迭代器
            last_output = output  # 更新最后一个输出

        if last_output["error_code"] != 0:  # 如果最后一个输出的错误代码不为 0
            return create_error_response(last_output["error_code"], last_output["text"])  # 返回创建错误响应

        logprobs = None
        if params.get("logprobs") and last_output["logprobs"]:  # 如果参数中有 logprobs 并且最后一个输出中有 logprobs
            logprobs = model_validate(Logprobs, last_output["logprobs"])  # 验证 logprobs

        choice = CompletionChoice(  # 创建完成选择对象
            index=0,
            text=last_output["text"],  # 完成的文本内容
            finish_reason="stop",  # 结束原因设为 "stop"
            logprobs=logprobs,  # logprobs
        )
        usage = model_validate(CompletionUsage, last_output["usage"])  # 验证完成使用情况
        return Completion(  # 返回完成对象
            id=last_output["id"],  # 输出的 ID
            choices=[choice],  # 完成选择对象列表
            created=last_output["created"],  # 创建时间
            model=last_output["model"],  # 模型名称
            object="text_completion",  # 对象类型
            usage=usage,  # 完成使用情况
        )

    def _create_chat_completion_stream(self, params: Dict[str, Any]) -> Iterator[ChatCompletionChunk]:
        """
        Creates a chat completion stream.

        Args:
            params (Dict[str, Any]): The parameters for generating the chat completion.  # 参数说明：生成聊天完成的参数字典

        Yields:
            Iterator[ChatCompletionChunk]: The output of the chat completion stream.  # 返回值说明：聊天完成流的输出
        """
        _id, _created, _model = None, None, None
        has_function_call = False
        for i, output in enumerate(self._generate(params)):  # 遍历生成文本的迭代器
            if output["error_code"] != 0:  # 如果输出的错误代码不为 0
                yield output  # 返回输出
                return  # 终止生成

            _id, _created, _model = output["id"], output["created"], output["model"]  # 更新 ID、创建时间和模型名称
            if i == 0:  # 对于第一个输出
                choice = ChunkChoice(  # 创建块选择对象
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=""),  # 完成的角色和内容
                    finish_reason=None,  # 结束原因设为 None
                    logprobs=None,  # logprobs 设为 None
                )
                yield ChatCompletionChunk(  # 返回聊天完成块对象
                    id=f"chat{_id}",  # 添加前缀的 ID
                    choices=[choice],  # 块选择对象列表
                    created=_created,  # 创建时间
                    model=_model,  # 模型名称
                    object="chat.completion.chunk",  # 对象类型
                )

            finish_reason = output["finish_reason"]  # 获取完成原因
            if len(output[
                       "delta"]) == 0 and finish_reason != "function_call":  # 如果输出的 delta 长度为 0 并且完成原因不是 "function_call"
                continue  # 继续下一次循环

            function_call = None
            if finish_reason == "function_call":  # 如果完成原因是 "function_call"
                try:
                    _, function_call = self.prompt_adapter.parse_assistant_response(  # 解析助手响应中的函数调用
                        output["text"], params.get("functions"), params.get("tools"),
                    )
                except Exception as e:
                    traceback.print_exc()  # 打印错误堆栈信息
                    logger.warning("Failed to parse tool call")  # 记录警告信息

            if isinstance(function_call, dict) and "arguments" in function_call:  # 如果函数调用是字典且包含 "arguments"
                has_function_call = True  # 设置存在函数调用标志为 True
                function_call = ChoiceDeltaFunctionCall(**function_call)  # 创建函数调用对象
                delta = ChoiceDelta(  # 创建选择增量对象
                    content=output["delta"],  # 完成的内容
                    function_call=function_call  # 函数调用对象
                )
            elif isinstance(function_call, dict) and "function" in function_call:  # 如果函数调用是字典且包含 "function"
                has_function_call = True  # 设置存在函数调用标志为 True
                finish_reason = "tool_calls"  # 完成原因设为 "tool_calls"
                function_call["index"] = 0  # 函数调用索引设为 0
                tool_calls = [model_validate(ChoiceDeltaToolCall, function_call)]  # 验证工具调用
                delta = ChoiceDelta(  # 创建选择增量对象
                    content=output["delta"],  # 完成的内容
                    tool_calls=tool_calls,  # 工具调用列表
                )
            else:
                delta = ChoiceDelta(content=output["delta"])  # 创建选择增量对象

            choice = ChunkChoice(  # 创建块选择对象
                index=0,
                delta=delta,
                finish_reason=finish_reason,
                logprobs=None,
            )
            yield ChatCompletionChunk(  # 返回聊天完成块对象
                id=f"chat{_id}",  # 添加前缀的 ID
                choices=[choice],  # 块选择对象列表
                created=_created,  # 创建时间
                model=_model,  # 模型名称
                object="chat.completion.chunk",  # 对象类型
            )

        if not has_function_call:  # 如果不存在函数调用
            choice = ChunkChoice(  # 创建块选择对象
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop",  # 结束原因设为 "stop"
                logprobs=None,
            )
            yield ChatCompletionChunk(  # 返回聊天完成块对象
                id=f"chat{_id}",  # 添加前缀的 ID
                choices=[choice],  # 块选择对象列表
                created=_created,  # 创建时间
                model=_model,  # 模型名称
                object="chat.completion.chunk",  # 对象类型
            )

    def _create_chat_completion(self, params: Dict[str, Any]) -> Union[ChatCompletion, JSONResponse]:
        """
        Creates a chat completion based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for generating the chat completion.  # 参数说明：生成聊天完成的参数字典

        Returns:
            Union[ChatCompletion, JSONResponse]: The generated chat completion or a JSON response in case of error.  # 返回值说明：生成的聊天完成对象或在错误情况下的 JSON 响应
        """
        last_output = None
        for output in self._generate(params):  # 遍历生成文本的迭代器
            last_output = output  # 更新最后一个输出

        if last_output["error_code"] != 0:  # 如果最后一个输出的错误代码不为 0
            return create_error_response(last_output["error_code"], last_output["text"])  # 返回创建错误响应

        function_call, finish_reason = None, "stop"  # 初始化函数调用和结束原因
        if params.get("functions") or params.get("tools"):  # 如果参数中有 functions 或 tools
            try:
                res, function_call = self.prompt_adapter.parse_assistant_response(  # 解析助手响应中的函数调用
                    last_output["text"], params.get("functions"), params.get("tools"),
                )
                last_output["text"] = res  # 更新最后输出的文本
            except Exception as e:
                traceback.print_exc()  # 打印错误堆栈信息
                logger.warning("Failed to parse tool call")  # 记录警告信息

        if isinstance(function_call, dict) and "arguments" in function_call:  # 如果函数调用是字典且包含 "arguments"
            finish_reason = "function_call"  # 设置结束原因为 "function_call"
            function_call = FunctionCall(**function_call)  # 创建函数调用对象
            message = ChatCompletionMessage(  # 创建聊天完成消息对象
                role="assistant",  # 角色设为 "assistant"
                content=last_output["text"],  # 完成的内容
                function_call=function_call,  # 函数调用对象
            )
        elif isinstance(function_call, dict) and "function" in function_call:  # 如果函数调用是字典且包含 "function"
            finish_reason = "tool_calls"  # 设置结束原因为 "tool_calls"
            tool_calls = [model_validate(ChatCompletionMessageToolCall, function_call)]  # 验证聊天完成消息工具调用
            message = ChatCompletionMessage(  # 创建聊天完成消息对象
                role="assistant",  # 角色设为 "assistant"
                content=last_output["text"],  # 完成的内容
                tool_calls=tool_calls,  # 工具调用列表
            )
        else:
            message = ChatCompletionMessage(  # 创建聊天完成消息对象
                role="assistant",  # 角色设为 "assistant"
                content=last_output["text"].strip(),  # 去除首尾空白的完成内容
            )

        choice = Choice(  # 创建选择对象
            index=0,
            message=message,  # 聊天完成消息对象
            finish_reason=finish_reason,  # 结束原因
            logprobs=None,  # logprobs 设为 None
        )
        usage = model_validate(CompletionUsage, last_output["usage"])  # 验证完成使用情况
        return ChatCompletion(  # 返回聊天完成对象
            id=f"chat{last_output['id']}",  # 添加前缀的 ID
            choices=[choice],  # 选择对象列表
            created=last_output["created"],  # 创建时间
            model=last_output["model"],  # 模型名称
            object="chat.completion",  # 对象类型
            usage=usage,  # 完成使用情况
        )

    def create_completion(
            self,
            params: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Union[Iterator[Completion], Completion]:
        params = params or {}  # 如果参数为空则初始化为空字典
        params.update(kwargs)  # 更新参数

        return (
            self._create_completion_stream(params)  # 如果设置了 'stream' 参数为 True，则返回生成的完成流迭代器
            if params.get("stream", False)  # 检查 'stream' 参数是否为 True
            else self._create_completion(params)  # 否则返回生成的单个完成对象
        )

    def create_chat_completion(
            self,
            params: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> Union[Iterator[ChatCompletionChunk], ChatCompletion]:
        params = params or {}  # 如果参数为空则初始化为空字典
        params.update(kwargs)  # 更新参数

        return (
            self._create_chat_completion_stream(params)  # 如果设置了 'stream' 参数为 True，则返回生成的聊天完成流迭代器
            if params.get("stream", False)  # 检查 'stream' 参数是否为 True
            else self._create_chat_completion(params)  # 否则返回生成的单个聊天完成对象
        )

    @property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            Any: The stop property of the prompt adapter, or None if it does not exist.
        """
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None

