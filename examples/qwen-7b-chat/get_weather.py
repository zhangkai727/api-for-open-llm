import json

from colorama import init, Fore
from loguru import logger
from openai import OpenAI

init(autoreset=True)


client = OpenAI(
    api_key="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",
    base_url="http://192.168.20.59:7891/v1/",
)


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """获取指定位置的当前天气信息"""
    weather_info = {
        "location": location,  # 存储位置信息
        "temperature": "72",  # 设置温度信息（示例值）
        "unit": unit,  # 存储温度单位信息
        "forecast": ["sunny", "windy"],  # 设置天气预报信息（示例值）
    }
    return json.dumps(weather_info)  # 返回天气信息的JSON字符串表示


functions = [
    {
        "name": "get_current_weather",  # 函数名
        "description": "获取指定位置的当前天气信息。",  # 函数描述
        "parameters": {
            "type": "object",  # 参数类型为对象
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市和州，例如 San Francisco, CA",  # 位置参数的描述
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},  # 温度单位参数的描述及可选值
            },
            "required": ["location"],  # location 参数为必需
        },
    }
]

available_functions = {
    "get_current_weather": get_current_weather,
}  # 可用函数字典，本例只有一个函数，但可以有多个


def run_conversation(query: str, stream=False, functions=None, max_retry=5):
    """
    运行对话生成器。

    Args:
        query (str): 用户查询的文本。
        stream (bool, optional): 是否使用流式处理。默认为 False。
        functions (list, optional): 可用函数列表。默认为 None。
        max_retry (int, optional): 最大重试次数。默认为 5。
    """
    # 初始化参数
    params = dict(model="qwen", messages=[{"role": "user", "content": query}], stream=stream)
    if functions:
        params["functions"] = functions

    # 调用对话生成 API
    response = client.chat.completions.create(**params)

    # 最大重试次数内进行处理
    for _ in range(max_retry):
        if not stream:
            # 非流式处理逻辑
            if response.choices[0].message.function_call:
                # 如果存在函数调用
                function_call = response.choices[0].message.function_call
                logger.info(f"Function Call Response: {function_call.model_dump()}")

                # 获取要调用的函数及参数
                function_to_call = available_functions[function_call.name]
                function_args = json.loads(function_call.arguments)

                # 调用函数并获取返回结果
                tool_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                logger.info(f"Tool Call Response: {tool_response}")

                # 将函数调用及其返回结果添加到参数中
                params["messages"].append(response.choices[0].message.model_dump(include={"role", "content", "function_call"}))
                params["messages"].append(
                    {
                        "role": "function",
                        "name": function_call.name,
                        "content": tool_response,
                    }
                )
            else:
                # 如果没有函数调用，直接返回最终回复
                reply = response.choices[0].message.content
                logger.info(f"Final Reply: \n{reply}")
                return

        else:
            # 流式处理逻辑
            output = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(Fore.BLUE + content, end="", flush=True)
                output += content

                if chunk.choices[0].finish_reason == "stop":
                    return

                elif chunk.choices[0].finish_reason == "function_call":
                    # 如果遇到函数调用
                    print("\n")

                    function_call = chunk.choices[0].delta.function_call
                    logger.info(f"Function Call Response: {function_call.model_dump()}")

                    # 获取要调用的函数及参数
                    function_to_call = available_functions[function_call.name]
                    function_args = json.loads(function_call.arguments)
                    tool_response = function_to_call(
                        location=function_args.get("location"),
                        unit=function_args.get("unit"),
                    )
                    logger.info(f"Tool Call Response: {tool_response}")

                    # 将函数调用及其返回结果添加到参数中
                    params["messages"].append(
                        {
                            "role": "assistant",
                            "content": output,
                            "function_call": function_call,
                        }
                    )
                    params["messages"].append(
                        {
                            "role": "function",
                            "name": function_call.name,
                            "content": tool_response,
                        }
                    )

                    break

        # 继续调用对话生成 API
        response = client.chat.completions.create(**params)


if __name__ == "__main__":
    # 示例查询：你是谁
    query = "你是谁"
    run_conversation(query, stream=False)

    logger.info("\n=========== next conversation ===========")

    # 示例查询：波士顿天气如何？
    query = "波士顿天气如何？"
    run_conversation(query, functions=functions, stream=False)

