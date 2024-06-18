import os

import streamlit as st
from openai import OpenAI

from .utils import CodeKernel, extract_code, execute, postprocess_text


<<<<<<< HEAD
@st.cache_resource  # 使用Streamlit的缓存装饰器缓存资源
def get_kernel():  # 定义获取代码内核的函数
    return CodeKernel()  # 返回一个CodeKernel实例


SYSTEM_MESSAGE = [  # 定义系统消息
=======
@st.cache_resource
def get_kernel():
    return CodeKernel()


SYSTEM_MESSAGE = [
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    {
        "role": "system",
        "content": "你是一位智能AI助手，你叫ChatGLM，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。"
    }
]


<<<<<<< HEAD
def chat_once(message_placeholder, client: OpenAI):  # 定义一次聊天的函数
    params = dict(  # 设置聊天参数
        model="chatglm3",  # 使用的模型
        messages=SYSTEM_MESSAGE + st.session_state.messages,  # 系统消息和会话消息
        stream=True,  # 启用流式响应
        max_tokens=st.session_state.get("max_tokens", 512),  # 最大令牌数
        temperature=st.session_state.get("temperature", 0.9),  # 温度参数
    )
    response = client.chat.completions.create(**params)  # 调用OpenAI API生成聊天回复

    display = ""  # 初始化显示字符串
    for _ in range(5):  # 重试5次
        full_response = ""  # 初始化完整响应字符串
        for chunk in response:  # 遍历响应流
            content = chunk.choices[0].delta.content or ""  # 获取响应内容
            full_response += content  # 拼接完整响应
            display += content  # 拼接显示内容
            message_placeholder.markdown(postprocess_text(display) + "▌")  # 显示当前生成的部分响应

            if chunk.choices[0].finish_reason == "stop":  # 如果响应完成
                message_placeholder.markdown(postprocess_text(display) + "▌")  # 显示完整响应
                st.session_state.messages.append(  # 将助手的完整响应添加到消息列表中
=======
def chat_once(message_placeholder, client: OpenAI):
    params = dict(
        model="chatglm3",
        messages=SYSTEM_MESSAGE + st.session_state.messages,
        stream=True,
        max_tokens=st.session_state.get("max_tokens", 512),
        temperature=st.session_state.get("temperature", 0.9),
    )
    response = client.chat.completions.create(**params)

    display = ""
    for _ in range(5):
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            display += content
            message_placeholder.markdown(postprocess_text(display) + "▌")

            if chunk.choices[0].finish_reason == "stop":
                message_placeholder.markdown(postprocess_text(display) + "▌")
                st.session_state.messages.append(
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                    {
                        "role": "assistant",
                        "content": full_response
                    }
                )
<<<<<<< HEAD
                return  # 返回

            elif chunk.choices[0].finish_reason == "function_call":  # 如果需要调用函数
                try:
                    code = extract_code(full_response)  # 提取代码
                except:
                    continue  # 继续下一个响应

                with message_placeholder:  # 显示代码执行状态
                    with st.spinner("Executing code..."):  # 显示执行代码的加载状态
                        try:
                            res_type, res = execute(code, get_kernel())  # 执行代码并获取结果
                        except Exception as e:
                            st.error(f"Error when executing code: {e}")  # 显示执行代码错误
                            return  # 返回

                if res_type == "text":  # 如果结果是文本
                    res = postprocess_text(res)  # 后处理文本
                    display += "\n" + res  # 拼接显示内容
                    message_placeholder.markdown(postprocess_text(display) + "▌")  # 显示当前生成的部分响应
                elif res_type == "image":  # 如果结果是图像
                    st.image(res)  # 显示图像

                st.session_state.messages.append(  # 将助手的完整响应添加到消息列表中
=======
                return

            elif chunk.choices[0].finish_reason == "function_call":
                try:
                    code = extract_code(full_response)
                except:
                    continue

                with message_placeholder:
                    with st.spinner("Executing code..."):
                        try:
                            res_type, res = execute(code, get_kernel())
                        except Exception as e:
                            st.error(f"Error when executing code: {e}")
                            return

                if res_type == "text":
                    res = postprocess_text(res)
                    display += "\n" + res
                    message_placeholder.markdown(postprocess_text(display) + "▌")
                elif res_type == "image":
                    st.image(res)

                st.session_state.messages.append(
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                    {
                        "role": "assistant",
                        "content": full_response,
                    }
                )
<<<<<<< HEAD
                st.session_state.messages.append(  # 将函数调用结果添加到消息列表中
=======
                st.session_state.messages.append(
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                    {
                        "role": "function",
                        "name": "interpreter",
                        "content": "[Image]" if res_type == "image" else res,  # 调用函数返回结果
                    }
                )

<<<<<<< HEAD
                break  # 结束当前响应处理

        params["messages"] = st.session_state.messages  # 更新消息上下文
        response = client.chat.completions.create(**params)  # 重新调用OpenAI API生成聊天回复


def main():  # 定义主函数
    st.title("💬 Code Interpreter")  # 设置页面标题为"💬 Code Interpreter"

    client = OpenAI(  # 创建OpenAI客户端实例
        api_key=os.getenv("API_KEY"),  # 从环境变量获取API密钥
        base_url=os.getenv("INTERPRETER_CHAT_API_BASE"),  # 从环境变量获取基础URL
    )

    if "messages" not in st.session_state:  # 如果会话状态中没有消息列表
        st.session_state.messages = []  # 初始化消息列表

    for message in st.session_state.messages:  # 遍历消息列表中的每条消息
        role = message["role"]  # 获取消息的角色
        if role in ["user", "function"]:  # 如果角色是用户或函数
            with st.chat_message("user"):  # 显示用户的聊天消息
                st.markdown(message["content"])  # 使用Markdown显示消息内容
        else:  # 否则（角色是助手）
            with st.chat_message("assistant"):  # 显示助手的聊天消息
                st.markdown(postprocess_text(message["content"]))  # 使用Markdown显示助手消息内容

    if prompt := st.chat_input("What is up?"):  # 如果用户在聊天输入框中输入了内容
        st.session_state.messages.append({"role": "user", "content": prompt})  # 将用户输入的内容添加到消息列表中
        with st.chat_message("user"):  # 显示用户的聊天消息
            st.markdown(prompt)  # 使用Markdown显示用户输入的内容

        with st.chat_message("assistant"):  # 显示助手的聊天消息
            message_placeholder = st.empty()  # 创建一个空的占位符用于显示助手的响应
            chat_once(message_placeholder, client)  # 调用一次聊天的函数


if __name__ == "__main__":  # 如果该脚本作为主程序执行
    main()  # 调用主函数

=======
                break

        params["messages"] = st.session_state.messages
        response = client.chat.completions.create(**params)


def main():
    st.title("💬 Code Interpreter")

    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("INTERPRETER_CHAT_API_BASE"),
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role = message["role"]
        if role in ["user", "function"]:
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(postprocess_text(message["content"]))

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            chat_once(message_placeholder, client)


if __name__ == "__main__":
    main()
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
