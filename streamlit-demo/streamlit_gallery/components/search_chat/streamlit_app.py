import os

import streamlit as st
from langchain_community.utilities.serpapi import SerpAPIWrapper
from openai import OpenAI

PROMPT_TEMPLATE = """<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>

<已知信息>问题的搜索结果为：{context}</已知信息>

<问题>{query}</问题>"""


def main():
<<<<<<< HEAD
    # 设置网页的标题
    st.title("💬 Search Chatbot")

    # 初始化 OpenAI 客户端，设置 API 密钥和基础 URL
=======
    st.title("💬 Search Chatbot")

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("CHAT_API_BASE"),
    )

<<<<<<< HEAD
    # 初始化 SerpAPIWrapper 实例，用于执行搜索
    search = SerpAPIWrapper()

    # 如果会话状态中没有存储消息，则初始化一个空的消息列表
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 遍历会话状态中的所有消息并显示出来
    for message in st.session_state.messages:
        # 根据消息的角色创建一个聊天消息组件
        with st.chat_message(message["role"]):
            # 显示消息的内容
            st.markdown(message["content"])
            # 如果消息的角色是 assistant 并且包含参考信息，则显示参考搜索结果
            if message["role"] == "assistant" and message["reference"] is not None:
                # 显示参考搜索结果的标题
                st.markdown("### Reference Search Results")
                # 以 JSON 格式显示参考搜索结果
                st.json(message["reference"], expanded=False)

    # 获取用户输入的聊天信息
    if prompt := st.chat_input("What is up?"):
        # 将用户的消息添加到会话状态中
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 显示用户的消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 创建一个聊天消息组件用于显示助手的消息
        with st.chat_message("assistant"):
            # 运行搜索功能并获取结果
            result = search.run(prompt)
            # 创建一个占位符组件用于动态显示助手的响应
            message_placeholder = st.empty()
            full_response = ""
            # 调用 OpenAI API 获取响应
            for response in client.chat.completions.create(
                model="baichuan",
                messages=[
                    # 将先前的所有消息添加到请求中
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages[:-1]
                ] + [
                    # 将用户当前的消息和搜索结果添加到请求中
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(query=prompt, context=result)
                    }
                ],
=======
    search = SerpAPIWrapper()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message["reference"] is not None:
                st.markdown("### Reference Search Results")
                st.json(message["reference"], expanded=False)

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = search.run(prompt)
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model="baichuan",
                messages=[
                     {
                         "role": m["role"],
                         "content": m["content"]
                     }
                     for m in st.session_state.messages[:-1]
                 ] + [
                     {
                         "role": "user",
                         "content": PROMPT_TEMPLATE.format(query=prompt, context=result)
                     }
                 ],
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                max_tokens=st.session_state.get("max_tokens", 512),
                temperature=st.session_state.get("temperature", 0.9),
                stream=True,
            ):
<<<<<<< HEAD
                # 累积 OpenAI API 返回的内容片段
                full_response += response.choices[0].delta.content or ""
                # 动态显示累积的内容
                message_placeholder.markdown(full_response + "▌")
            # 显示完整的响应内容
            message_placeholder.markdown(full_response)

            # 显示参考搜索结果的标题
            st.markdown("### Reference Search Results")
            # 以 JSON 格式显示搜索结果
            st.json({"search_result": result}, expanded=False)

        # 将助手的消息及参考信息添加到会话状态中
=======
                full_response += response.choices[0].delta.content or ""

                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.markdown("### Reference Search Results")
            st.json({"search_result": result}, expanded=False)

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "reference": {"search_result": result},
            }
        )

<<<<<<< HEAD
# 如果脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()

=======

if __name__ == "__main__":
    main()
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
