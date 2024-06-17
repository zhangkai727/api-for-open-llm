import gc
import re
import time
import uuid
from typing import (
    List,
    Union,
    Dict,
    Any,
    Iterator,
)

import torch
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.generation.logits_process import LogitsProcessor

from api.generation.utils import apply_stopping_strings
from api.utils.protocol import Role


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            # 如果分数中存在 NaN 或者 Inf
            scores.zero_()
            # 将所有分数置零
            scores[..., 5] = 5e4
            # 将第五个位置的分数设置为 5e4
        return scores



def process_response(response: str) -> str:
    """
    处理响应，包括去除前后空白、替换训练时间占位符，并标准化标点符号。

    Args:
        response: 输入的响应字符串。

    Returns:
        处理后的响应字符串。
    """
    response = response.strip()
    # 去除前后空白

    response = response.replace("[[训练时间]]", "2023年")
    # 替换训练时间占位符为具体日期

    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    # 定义中英标点的对应关系列表

    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    # 使用正则表达式替换标点符号

    return response



def check_is_chatglm(model) -> bool:
    """
    检查给定的模型是否为 ChatGLM 模型。

    Args:
        model: 要检查的模型。

    Returns:
        bool: 如果模型是 ChatGLM 模型则返回 True，否则返回 False。
    """
    return "GLMBlock" in getattr(model, "_no_split_modules", [])
    # 检查模型是否具有 "_no_split_modules" 属性，并判断列表中是否包含 "GLMBlock"，返回布尔值结果



@torch.inference_mode()
def generate_stream_chatglm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: Dict[str, Any],
) -> Iterator:
    """
    使用 ChatGLM 模型以流式方式生成文本。

    Args:
        model: 预训练的 ChatGLM 模型。
        tokenizer: 用于分词输入的分词器。
        params: 包含输入参数的字典。

    Yields:
        每个生成的文本完成项的字典表示。

    """
    inputs = params["inputs"]  # 获取输入参数中的输入
    model_name = params.get("model", "llm")  # 获取模型名称，默认为 "llm"
    temperature = float(params.get("temperature", 1.0))  # 获取温度参数，默认为 1.0
    repetition_penalty = float(params.get("repetition_penalty", 1.0))  # 获取重复惩罚参数，默认为 1.0
    top_p = float(params.get("top_p", 1.0))  # 获取 top-p 参数，默认为 1.0
    max_new_tokens = int(params.get("max_tokens", 256))  # 获取生成的最大令牌数，默认为 256
    echo = params.get("echo", True)  # 获取回显标志，默认为 True

    input_echo_len = len(inputs["input_ids"][0])  # 获取输入的令牌数
    if input_echo_len >= model.config.seq_length:
        logger.warning(f"Input length larger than {model.config.seq_length}")  # 如果输入长度超过模型配置的最大序列长度，记录警告信息

    inputs = {k: v[:, -model.config.seq_length:].to(model.device) for k, v in inputs.items()}  # 将输入截断到模型的最大序列长度，并移动到设备上

    gen_kwargs = {
        "max_length": min(max_new_tokens + input_echo_len, model.config.seq_length),  # 设置生成的最大长度
        "do_sample": temperature > 1e-5,  # 是否进行采样
        "top_p": top_p,  # 设置 top-p 参数
        "repetition_penalty": repetition_penalty,  # 设置重复惩罚参数
        "logits_processor": [InvalidScoreLogitsProcessor()],  # 使用自定义的 logits 处理器
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature  # 设置温度参数

    total_len, previous_text = 0, ""  # 初始化生成的总长度和前一个文本
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"  # 创建生成完成的唯一标识符
    created: int = int(time.time())  # 记录生成的时间戳
    for total_ids in model.stream_generate(**inputs, **gen_kwargs):
        total_ids = total_ids.tolist()[0]  # 将生成的令牌转换为列表
        total_len = len(total_ids)  # 获取生成的总长度

        output_ids = total_ids if echo else total_ids[input_echo_len:]  # 根据回显标志选择是否输出所有生成的令牌
        response = tokenizer.decode(output_ids)  # 解码生成的令牌为文本
        response = process_response(response)  # 处理生成的响应文本

        delta_text = response[len(previous_text):]  # 计算变化的文本部分
        previous_text = response  # 更新前一个文本为当前响应文本

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "delta": delta_text,
            "text": response,
            "logprobs": None,
            "finish_reason": None,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
        }
        # 生成并返回包含生成文本相关信息的字典

    # 生成最后一个流式结果时，设置 finish_reason 为 stop
    yield {
        "id": completion_id,  # 生成的文本完成项的唯一标识符
        "object": "text_completion",  # 表示对象为文本完成项
        "created": created,  # 生成的时间戳
        "model": model_name,  # 使用的模型名称
        "delta": "",  # 文本变化部分为空字符串
        "text": response,  # 生成的文本
        "logprobs": None,  # logprobs 为空
        "finish_reason": "stop",  # 完成原因设置为 stop
        "usage": {
            "prompt_tokens": input_echo_len,  # 使用的提示令牌数
            "completion_tokens": total_len - input_echo_len,  # 生成的完成令牌数
            "total_tokens": total_len,  # 总共生成的令牌数
        },
    }
gc.collect()  # 手动触发垃圾回收
torch.cuda.empty_cache()  # 清空 CUDA 缓存


@torch.inference_mode()
def generate_stream_chatglm_v3(
    model: PreTrainedModel,  # 预训练的 ChatGLM 模型
    tokenizer: PreTrainedTokenizer,  # 用于标记化输入的分词器
    params: Dict[str, Any],  # 包含输入参数的字典

) -> Iterator:  # 生成器函数，每次产生一个文本完成项的字典
    """
    Generates text in a streaming manner using the ChatGLM model.

    Args:
        model: The pre-trained ChatGLM model.
        tokenizer: The tokenizer used for tokenizing the input.
        params: A dictionary containing the input parameters.

    Yields:
        A dictionary representing each generated text completion.

    """
    inputs = params["inputs"]  # 获取输入参数中的 inputs，包含 input_ids 和 attention_mask
    model_name = params.get("model", "llm")  # 获取模型名称，默认为 "llm"
    temperature = float(params.get("temperature", 1.0))  # 获取温度参数，默认为 1.0
    repetition_penalty = float(params.get("repetition_penalty", 1.0))  # 获取重复惩罚参数，默认为 1.0
    top_p = float(params.get("top_p", 1.0))  # 获取 top-p 参数，默认为 1.0
    max_new_tokens = int(params.get("max_tokens", 256))  # 获取最大新令牌数，默认为 256
    echo = params.get("echo", True)  # 获取 echo 参数，默认为 True，表示是否回显输入部分

    input_echo_len = len(inputs["input_ids"][0])  # 计算输入的 echo 长度
    if input_echo_len >= model.config.seq_length:  # 如果 echo 长度超过模型的序列长度
        logger.warning(f"Input length larger than {model.config.seq_length}")  # 记录警告日志

    inputs = {k: v[:, -model.config.seq_length:].to(model.device) for k, v in inputs.items()}  # 截断输入以适应模型长度，并移动到设备

    eos_token_id = [  # EOS 标记的 ID 列表
        tokenizer.eos_token_id,  # 默认的 EOS 标记 ID
        tokenizer.get_command(""),  # 获取空命令的标记 ID
    ]

    gen_kwargs = {  # 生成文本的参数字典
        "max_length": min(max_new_tokens + input_echo_len, model.config.seq_length),  # 最大生成长度
        "do_sample": temperature > 1e-5,  # 是否采样生成
        "top_p": top_p,  # top-p 参数
        "repetition_penalty": repetition_penalty,  # 重复惩罚参数
        "logits_processor": [InvalidScoreLogitsProcessor()],  # 使用自定义 logprobs 处理器
    }
    if temperature > 1e-5:  # 如果温度大于 1e-5
        gen_kwargs["temperature"] = temperature  # 添加温度参数到生成参数中

    total_len, previous_text = 0, ""  # 总长度和之前的文本初始化为空
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"  # 生成的文本完成项的唯一标识符
    created: int = int(time.time())  # 当前时间的时间戳
    for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):  # 在流生成过程中迭代生成的令牌 IDs
        total_ids = total_ids.tolist()[0]  # 将生成的 token IDs 转换为列表形式
        total_len = len(total_ids)  # 计算生成的总令牌数

        output_ids = total_ids[:-1] if echo else total_ids[input_echo_len:-1]  # 根据 echo 参数选择是否包含输入部分，同时去除 EOS 符号
        response = tokenizer.decode(output_ids)  # 解码生成的 token IDs 得到文本响应

        if response and response[-1] != "�":  # 检查响应是否以损坏符号结尾
            response, stop_found = apply_stopping_strings(response, [""])  # 应用停止字符串，检查是否包含停止信号

            delta_text = response[len(previous_text):]  # 计算新生成文本与之前文本的差异部分
            previous_text = response  # 更新之前文本为当前响应

            yield {
                "id": completion_id,  # 生成的文本完成项的唯一标识符
                "object": "text_completion",  # 表示对象为文本完成项
                "created": created,  # 生成的时间戳
                "model": model_name,  # 使用的模型名称
                "delta": delta_text,  # 文本变化部分
                "text": response,  # 生成的文本
                "logprobs": None,  # logprobs 为空
                "finish_reason": "function_call" if stop_found else None,  # 如果发现停止信号，则设置完成原因为 "function_call"
                "usage": {
                    "prompt_tokens": input_echo_len,  # 使用的提示令牌数
                    "completion_tokens": total_len - input_echo_len,  # 生成的完成令牌数
                    "total_tokens": total_len,  # 总共生成的令牌数
                },
            }

            if stop_found:
                break  # 如果发现停止信号，结束生成过程

        # 仅在最后一个流生成结果中包含 finish_reason，我们将 finish_reason 设置为 "stop"
        yield {
            "id": completion_id,  # 生成的文本完成项的唯一标识符
            "object": "text_completion",  # 表示对象为文本完成项
            "created": created,  # 生成的时间戳
            "model": model_name,  # 使用的模型名称
            "delta": "",  # 没有变化部分
            "text": response,  # 最后生成的文本
            "logprobs": None,  # logprobs 为空
            "finish_reason": "stop",  # 完成原因设置为 "stop"
            "usage": {
                "prompt_tokens": input_echo_len,  # 使用的提示令牌数
                "completion_tokens": total_len - input_echo_len,  # 生成的完成令牌数
                "total_tokens": total_len,  # 总共生成的令牌数
            },
        }

        gc.collect()  # 执行垃圾回收
        torch.cuda.empty_cache()  # 清空 CUDA 缓存


def process_chatglm_messages(
    messages: List[ChatCompletionMessageParam],
    functions: Union[dict, List[dict]] = None,
) -> List[dict]:
    """
    Processes a list of chat messages and returns a modified list of messages.

    Args:
        messages: A list of chat messages to be processed.
        functions: Optional. A dictionary or list of dictionaries representing the available tools.

    Returns:
        A modified list of chat messages.
    """
    _messages = messages
    messages = []

    if functions:  # 如果提供了函数（工具）
        messages.append(
            {
                "role": Role.SYSTEM,  # 系统角色
                "content": "Answer the following questions as best as you can. You have access to the following tools:",
                # 提示内容
                "tools": functions  # 工具信息
            }
        )  # 将系统提示信息添加到消息列表中

    for m in _messages:  # 遍历原始消息列表
        role, content = m["role"], m["content"]  # 获取消息的角色和内容
        if role == Role.FUNCTION:  # 如果角色是功能角色
            messages.append({"role": "observation", "content": content})  # 添加一个观察角色的消息
        elif role == Role.ASSISTANT:  # 如果角色是助手角色
            for response in content.split(""):  # 按空格分割内容
                if "\n" in response:  # 如果内容中包含换行符
                    metadata, sub_content = response.split("\n", maxsplit=1)  # 拆分元数据和子内容
                else:
                    metadata, sub_content = "", response  # 否则将元数据置空，内容为整个响应
                messages.append(
                    {"role": role, "metadata": metadata, "content": sub_content.strip()})  # 添加角色、元数据和清理后的内容到消息列表
        else:  # 其他角色
            messages.append({"role": role, "content": content})  # 直接将角色和内容添加到消息列表

    return messages  # 返回处理后的消息列表

