from typing import List, Tuple

from openai.types.chat import ChatCompletionMessageParam
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from api.utils.protocol import Role


def parse_messages(
    messages: List[ChatCompletionMessageParam], split_role=Role.USER
) -> Tuple[str, List[List[ChatCompletionMessageParam]]]:
    """
    Parse a list of chat completion messages into system and rounds.

    Args:
        messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
        split_role: The role at which to split the rounds. Defaults to Role.USER.

    Returns:
        Tuple[str, List[List[ChatCompletionMessageParam]]]: A tuple containing the system message and a list of rounds.
    """
    system, rounds = "", []  # 初始化系统消息和回合列表
    r = []  # 初始化单个回合的消息列表
    for i, message in enumerate(messages):
        if message["role"] == Role.SYSTEM:  # 如果消息角色为系统
            system = message["content"]  # 设置系统消息内容
            continue
        if message["role"] == split_role and r:  # 如果消息角色为指定分割角色且当前回合非空
            rounds.append(r)  # 将当前回合消息列表添加到回合列表中
            r = []  # 重置当前回合消息列表
        r.append(message)  # 添加消息到当前回合消息列表
    if r:  # 处理最后一个回合
        rounds.append(r)  # 将最后一个回合消息列表添加到回合列表中
    return system, rounds


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    """
    Prepare a list of logits processors based on the provided parameters.

    Args:
        temperature (float): The temperature value for temperature warping.
        repetition_penalty (float): The repetition penalty value.
        top_p (float): The top-p value for top-p warping.
        top_k (int): The top-k value for top-k warping.

    Returns:
        LogitsProcessorList: A list of logits processors.
    """
    processor_list = LogitsProcessorList()  # 初始化一个空的LogitsProcessorList对象，用于存储处理器

    # 添加温度调整处理器，如果温度大于等于1e-5且不等于1.0（因为1.0时TemperatureLogitsWarper是无操作）
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))

    # 添加重复惩罚处理器，如果重复惩罚大于1.0
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

    # 添加Top-P截断处理器，如果0.0 < top_p < 1.0
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))

    # 添加Top-K截断处理器，如果top_k大于0
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))

    return processor_list  # 返回填充了相应处理器的处理器列表



def is_partial_stop(output: str, stop_str: str):
    """ Check whether the output contains a partial stop str. """
    return any(
        stop_str.startswith(output[-i:])  # 检查stop_str是否以output的末尾子串开始
        for i in range(0, min(len(output), len(stop_str)))  # 遍历output和stop_str长度的最小值作为范围
    )


# Models don't use the same configuration key for determining the maximum
# sequence length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important.  Some models have two of these, and we
# have a preference for which value gets used.
SEQUENCE_LENGTH_KEYS = [
    "max_sequence_length",      # 最大序列长度的键名。
    "seq_length",               # 序列长度的键名。
    "max_position_embeddings",  # 最大位置嵌入的键名。
    "max_seq_len",              # 最大序列长度的备选键名。
    "model_max_length",         # 模型最大长度的键名。
]


def get_context_length(config) -> int:
    rope_scaling = getattr(config, "rope_scaling", None)  # 获取配置对象中的rope_scaling属性。
    rope_scaling_factor = config.rope_scaling["factor"] if rope_scaling else 1  # 获取rope_scaling的缩放因子，如果没有则默认为1。
    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)  # 从配置对象中获取与SEQUENCE_LENGTH_KEYS中键名对应的属性值。
        if val is not None:
            return int(rope_scaling_factor * val)  # 计算并返回经过缩放因子处理后的序列长度。
    return 2048  # 如果所有键都未找到对应的值，则返回默认的上下文长度2048。


def apply_stopping_strings(reply: str, stop_strings: List[str]) -> Tuple[str, bool]:
    """
    将停止字符串应用到回复中，并检查是否找到停止字符串。

    Args:
        reply (str): 要应用停止字符串的回复。
        stop_strings (List[str]): 要检查的停止字符串列表。

    Returns:
        Tuple[str, bool]: 包含修改后的回复和布尔值的元组，指示是否找到停止字符串。
    """
    stop_found = False  # 初始化停止字符串是否找到的标志为False。

    # 遍历停止字符串列表，查找第一个出现的停止字符串。
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]  # 截取找到停止字符串之前的部分作为修改后的回复。
            stop_found = True  # 设置停止字符串找到的标志为True。
            break  # 停止查找其他停止字符串。

    # 如果未找到停止字符串，尝试修剪可能存在的截断字符串。
    if not stop_found:
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:  # 检查回复末尾是否匹配截断字符串的开头。
                    reply = reply[:-j]  # 截断回复末尾与截断字符串开头匹配部分。
                    break
            else:
                continue

            break

    return reply, stop_found  # 返回修改后的回复和停止字符串是否找到的布尔值。

