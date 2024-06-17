from typing import List

from openai.types.chat import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer

from api.generation.utils import parse_messages
from api.utils.protocol import Role


def build_xverse_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatCompletionMessageParam],
    context_len: int = 8192,
    max_new_tokens: int = 256
) -> List[int]:
    """
    Builds the input tokens for the Xverse chat model based on the given messages.

    Refs:
        https://huggingface.co/xverse/XVERSE-13B-Chat/blob/main/modeling_xverse.py

    Args:
        tokenizer: The PreTrainedTokenizer object.
        messages: A list of ChatCompletionMessageParam objects representing the chat messages.
        context_len: The maximum length of the context (default=8192).
        max_new_tokens: The maximum number of new tokens to be added (default=256).

    Returns:
        List[int]: The input tokens for the Baichuan chat model.
    """
    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)  # 解析聊天消息，获取系统消息和回合消息列表
    system = f"{system}\n\n" if system else system  # 如果有系统消息，则在系统消息末尾添加两个换行符

    def _tokenize_str(role, content):
        """ 将角色和内容组合成字符串，并使用分词器编码成token列表。 """
        return tokenizer.encode(f"{role}: {content}", return_token_type_ids=False)

    system_tokens = tokenizer.encode(system, return_token_type_ids=False)  # 使用分词器编码系统消息成token列表
    max_history_tokens = max_input_tokens - len(system_tokens)  # 计算可用于历史消息的最大token数量

    history_tokens = []
    for i, r in enumerate(rounds[::-1]):
        round_tokens = []
        for message in r:
            if message["role"] == Role.USER:
                content = f"{message['content']}\n\n"  # 用户角色的内容后面加两个换行符
                if i == 0:
                    content += "Assistant: "  # 如果是第一个回合，添加“Assistant: ”作为开头
                content_tokens = _tokenize_str("Human", content)  # 使用分词器编码用户角色的内容
            else:
                content_tokens = _tokenize_str("Assistant", f"{message['content']}") + [3]  # 添加EOS token id作为结束标志

            round_tokens.extend(content_tokens)  # 将编码后的token列表扩展到当前回合的token列表中

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # 将当前回合的token列表拼接到历史token列表的左侧
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens  # 将系统消息token列表和历史消息token列表拼接成输入token列表
    return input_tokens[-max_input_tokens:]  # 返回截取左侧的输入token列表，确保不超过最大输入token数量


def check_is_xverse(model) -> bool:
    """
    Checks if the given model is a Xverse model.

    Args:
        model: The model to be checked.

    Returns:
        bool: True if the model is a Xverse model, False otherwise.
    """
    return "XverseDecoderLayer" in getattr(model, "_no_split_modules", [])
