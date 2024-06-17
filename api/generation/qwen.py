import json
from copy import deepcopy
from typing import (
    List,
    Union,
    Optional,
    Dict,
    Any,
    Tuple,
)

from loguru import logger
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from transformers import PreTrainedTokenizer

from api.utils.protocol import Role

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def build_qwen_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatCompletionMessageParam],
    max_window_size: int = 6144,
    functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """
    Builds the input tokens for Qwen chat generation.

    Refs:
        https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py

    Args:
        tokenizer: The tokenizer used to encode the input tokens.
        messages: The list of chat messages.
        max_window_size: The maximum length of the context.
        functions: Optional dictionary or list of dictionaries representing the functions.
        tools: Optional list of dictionaries representing the tools.

    Returns:
        The list of input tokens.
    """
    query, history, system = process_qwen_messages(messages, functions, tools)
    # 调用函数处理Qwen消息，获取查询、历史和系统消息。

    if query is _TEXT_COMPLETION_CMD:
        return build_last_message_input(tokenizer, history, system)
    # 如果查询是_TEXT_COMPLETION_CMD，返回最后消息的输入。

    im_start_tokens, im_end_tokens = [tokenizer.im_start_id], [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")
    # 初始化图像起始和结束token以及换行token。

    if hasattr(tokenizer, "IMAGE_ST"):
        def _tokenize_str(role, content):
            return tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))
    else:
        def _tokenize_str(role, content):
            return tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())
    # 根据tokenizer是否具有IMAGE_ST属性，定义函数_tokenize_str以将角色和内容编码为token序列。

    system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
    # 使用系统消息构建系统token序列。

    context_tokens = []
    # 初始化上下文token序列。

    for turn_query, turn_response in reversed(history):
        query_tokens_part = _tokenize_str("user", turn_query)
        query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

        response_tokens_part = _tokenize_str("assistant", turn_response)
        response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

        next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
        # 构建每个用户查询和助手响应的token序列。

        current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
        )
        # 计算当前上下文的总token数。

        if current_context_size < max_window_size:
            context_tokens = next_context_tokens + context_tokens
        else:
            break
        # 如果当前上下文token数小于最大窗口大小，将下一个上下文添加到上下文token序列中。

    context_tokens = system_tokens + context_tokens
    # 将系统token序列添加到上下文token序列中。

    context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
    )
    # 将查询和助手信息编码为token序列，并添加到上下文token序列中。

    return context_tokens
    # 返回构建的Qwen聊天输入的token序列。


def check_is_qwen(model) -> bool:
    """
    Checks if the given model is a Qwen model.

    Args:
        model: The model to be checked.

    Returns:
        bool: True if the model is a Qwen model, False otherwise.
    """
    return "QWenBlock" in getattr(model, "_no_split_modules", [])
# 检查模型是否是Qwen模型，通过检查模型的_no_split_modules属性中是否包含"QWenBlock"字符串来判断。


def process_qwen_messages(
    messages: List[ChatCompletionMessageParam],  # 表示要处理的聊天消息的ChatCompletionMessageParam对象列表。
    functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,  # 可选的字典或字典列表，表示可用的函数。
    tools: Optional[List[Dict[str, Any]]] = None,  # 可选的字典列表，表示在处理过程中要使用的工具。
) -> Tuple[str, List[List[str]], str]:  # 返回一个元组，包含从消息中提取的查询、历史和系统部分。

    """
    Process the Qwen messages and generate a query and history.

    Args:
        messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
        functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]): The functions to be used.
        tools (Optional[List[Dict[str, Any]]]): The tools to be used.

    Returns:
        Tuple[str, List[List[str]], str]: The generated query and history and system.
    """
    if all(m["role"] != Role.USER for m in messages):
        raise ValueError(f"Invalid messages: Expecting at least one user message.")
        # 如果所有消息的角色都不是用户，则引发值错误，期望至少有一条用户消息。

    messages = deepcopy(messages)
    if messages[0]["role"] == Role.SYSTEM:
        system = messages.pop(0)["content"].lstrip("\n").rstrip()
    else:
        system = "You are a helpful assistant."
    # 如果第一条消息的角色是系统，则将系统内容提取出来；否则默认系统消息为"You are a helpful assistant"。

    if tools:
        functions = [t["function"] for t in tools]

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get("name", "")
            name_m = func_info.get("name_for_model", name)
            name_h = func_info.get("name_for_human", name)
            desc = func_info.get("description", "")
            desc_m = func_info.get("description_for_model", desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )

            tools_text.append(tool)
            tools_name_text.append(name_m)

        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        instruction = REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        ).lstrip('\n').rstrip()
    else:
        instruction = ""
    # 如果存在工具，则生成工具的描述信息和指令；否则指令为空。

    messages_with_fncall = messages
    messages = []
    for m_idx, m in enumerate(messages_with_fncall):
        role, content = m["role"], m["content"]
        func_call, tool_calls = m.get("function_call", None), m.get("tool_calls", None)

        content = content or ''
        content = content.lstrip('\n').rstrip()

        if role in [Role.FUNCTION, Role.TOOL]:
            if (len(messages) == 0) or (messages[-1]["role"] != Role.ASSISTANT):
                raise ValueError(f"Invalid messages: Expecting role assistant before role function.")
            # 如果当前消息是函数或工具调用，并且前一条消息不是助手角色，则引发值错误，期望前一条消息是助手角色。

            messages[-1]["content"] += f"\nObservation: {content}"
            if m_idx == len(messages_with_fncall) - 1:
                messages[-1]["content"] += "\nThought:"
            # 将观察内容附加到前一条消息的内容中，并在最后一条消息之后加上思考标记。

        elif role == Role.ASSISTANT:
            if len(messages) == 0:
                raise ValueError(f"Invalid messages: Expecting role user before role assistant.")
            # 如果当前消息是助手角色，并且前面没有用户角色的消息，则引发值错误，期望前面有用户角色的消息。

            if func_call is None and tool_calls is None:
                if functions or tool_calls:
                    content = f"Thought: I now know the final answer.\nFinal Answer: {content}"
            # 如果没有函数或工具调用，则添加助手的思考和最终答案。

            if messages[-1]["role"] in [Role.USER, Role.SYSTEM]:
                messages.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content=content.lstrip("\n").rstrip())
                )
            else:
                messages[-1]["content"] += content
            # 如果前一条消息是用户或系统角色，则直接添加新的助手消息；否则将当前内容附加到前一条助手消息的内容中。

        elif role in [Role.USER, Role.SYSTEM]:
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=content.lstrip("\n").rstrip())
            )
    # 如果当前消息是用户或系统角色，则直接添加新的用户消息。

    query = _TEXT_COMPLETION_CMD  # 默认查询为_TEXT_COMPLETION_CMD

    if messages[-1]["role"] == Role.USER:
        query = messages[-1]["content"]  # 如果最后一条消息是用户角色，则将其内容作为查询内容
        messages = messages[:-1]  # 移除最后一条消息

    if len(messages) % 2 != 0:
        raise ValueError("Invalid messages")  # 如果消息数目为奇数，则引发值错误

    history = []  # 历史对话记录，格式为[(用户问题1, 助手回答1), (用户问题2, 助手回答2), ..., (用户最后一轮问题, 助手最后一轮回答)]
    for i in range(0, len(messages), 2):
        if messages[i]["role"] == Role.USER and messages[i + 1]["role"] == Role.ASSISTANT:
            usr_msg = messages[i]["content"].lstrip("\n").rstrip()  # 用户消息内容去除左右空格和换行符
            bot_msg = messages[i + 1]["content"].lstrip("\n").rstrip()  # 助手消息内容去除左右空格和换行符
            if instruction and (i == len(messages) - 2):
                usr_msg = f"{instruction}\n\nQuestion: {usr_msg}"  # 如果存在指令且当前消息是倒数第二条消息，则添加指令和问题标记到用户消息中
                instruction = ''  # 清空指令

            history.append([usr_msg, bot_msg])  # 将用户消息和助手消息作为一对添加到历史记录中
        else:
            raise ValueError(
                "Invalid messages: Expecting exactly one user (or function) role before every assistant role.")
            # 如果消息角色不符合预期，则引发值错误

    if instruction:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{instruction}\n\nQuestion: {query}"  # 如果存在指令且查询不是_TEXT_COMPLETION_CMD，则添加指令和问题标记到查询中

    return query, history, system  # 返回查询内容、历史对话记录和系统消息


def build_last_message_input(tokenizer: PreTrainedTokenizer, history: List[List[str]], system: str):
    im_start = ""  # 初始化空的开头标记
    im_end = ""  # 初始化空的结束标记
    prompt = f"{im_start}system\n{system}{im_end}"  # 初始化提示文本，包含系统消息

    for i, (query, response) in enumerate(history):
        query = query.lstrip("\n").rstrip()  # 去除查询内容的左侧换行符和右侧空格
        response = response.lstrip("\n").rstrip()  # 去除回复内容的左侧换行符和右侧空格
        prompt += f"\n{im_start}user\n{query}{im_end}"  # 添加用户查询到提示文本中
        prompt += f"\n{im_start}assistant\n{response}{im_end}"  # 添加助手回复到提示文本中

    prompt = prompt[:-len(im_end)]  # 移除最后一个结束标记
    logger.debug(f"==== Prompt with tools ====\n{prompt}")  # 记录调试信息，显示带有工具的提示文本
    return tokenizer.encode(prompt)  # 使用分词器编码最终的提示文本并返回

