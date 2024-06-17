from typing import List

from openai.types.chat import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer

from api.generation.utils import parse_messages
from api.utils.protocol import Role


def build_baichuan_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatCompletionMessageParam],
    context_len: int = 4096,
    max_new_tokens: int = 256
) -> List[int]:
    """
    Builds the input tokens for the Baichuan chat model based on the given messages.

    Refs:
        https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_utils.py 

    Args:
        tokenizer: The PreTrainedTokenizer object.
        messages: A list of ChatCompletionMessageParam objects representing the chat messages.
        context_len: The maximum length of the context (default=4096).
        max_new_tokens: The maximum number of new tokens to be added (default=256).

    Returns:
        List[int]: The input tokens for the Baichuan chat model.
    """
    max_input_tokens = context_len - max_new_tokens
    # 计算允许的最大历史 token 数量

    system, rounds = parse_messages(messages)
    # 解析消息以获取系统和轮数信息

    system_tokens = tokenizer.encode(system)
    # 使用分词器编码系统信息为 token IDs

    max_history_tokens = max_input_tokens - len(system_tokens)
    # 计算允许的最大历史 token 数量，减去系统 token 数量后的剩余空间

    history_tokens = []
    # 初始化历史 token 列表

    for r in rounds[::-1]:
        # 逆序遍历每个轮次的消息
        round_tokens = []
        for message in r:
            # 遍历每条消息
            if message["role"] == Role.USER:
                round_tokens.append(195)  # 标记用户消息
            else:
                round_tokens.append(196)  # 标记助理消息
            round_tokens.extend(tokenizer.encode(message["content"]))
            # 编码消息内容为 token IDs 并添加到当前轮次的 token 列表中

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            # 如果历史 token 列表为空或加上当前轮次的 token 数量不超过最大允许的历史 token 数量
            history_tokens = round_tokens + history_tokens  # 合并到历史 token 列表中（从左侧添加）
            if len(history_tokens) < max_history_tokens:
                continue
        break
        # 如果已经达到最大历史 token 数量则停止添加历史 token

    input_tokens = system_tokens + history_tokens
    # 组合系统 token 和历史 token 成为输入 token 列表

    if messages[-1]["role"] != Role.ASSISTANT:
        input_tokens.append(196)
        # 如果最后一条消息的角色不是助理，则添加助理标记到输入 token 列表的末尾

    return input_tokens[-max_input_tokens:]
    # 返回截断后的输入 token 列表（从左侧截断）


def check_is_baichuan(model) -> bool:
    """
    Checks if the given model is a Baichuan model.

    Args:
        model: The model to be checked.

    Returns:
        bool: True if the model is a Baichuan model, False otherwise.
    """
    return "BaichuanLayer" in getattr(model, "_no_split_modules", [])
