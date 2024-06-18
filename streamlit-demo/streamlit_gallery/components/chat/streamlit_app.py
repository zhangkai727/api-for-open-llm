import streamlit as st
from openai import OpenAI


<<<<<<< HEAD
def main():  # 定义主函数
    st.title("💬 Chatbot")  # 设置页面标题为"💬 Chatbot"

    client = OpenAI(  # 创建OpenAI客户端实例
        api_key=st.session_state.get("api_key", "xxx"),  # 从会话状态获取API密钥
        base_url=st.session_state.get("base_url", "xxx"),  # 从会话状态获取基础URL
    )

    if "messages" not in st.session_state:  # 如果会话状态中没有消息列表
        st.session_state.messages = []  # 初始化消息列表

    for message in st.session_state.messages:  # 遍历消息列表中的每条消息
        with st.chat_message(message["role"]):  # 根据消息的角色显示聊天消息
            st.markdown(message["content"])  # 使用Markdown显示消息内容

    if prompt := st.chat_input("What is up?"):  # 如果用户在聊天输入框中输入了内容
        st.session_state.messages.append({"role": "user", "content": prompt})  # 将用户输入的内容添加到消息列表中
        with st.chat_message("user"):  # 显示用户的聊天消息
            st.markdown(prompt)  # 使用Markdown显示用户输入的内容

        with st.chat_message("assistant"):  # 显示助手的聊天消息
            message_placeholder = st.empty()  # 创建一个空的占位符用于显示助手的响应
            full_response = ""  # 初始化完整响应字符串
            for response in client.chat.completions.create(  # 调用OpenAI API生成聊天回复
                model=st.session_state.get("model_name", "xxx"),  # 获取使用的模型名称
                messages=[  # 构建消息上下文列表
=======
def main():
    st.title("💬 Chatbot")

    client = OpenAI(
        api_key=st.session_state.get("api_key", "xxx"),
        base_url=st.session_state.get("base_url", "xxx"),
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state.get("model_name", "xxx"),
                messages=[
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages
                ],
<<<<<<< HEAD
                max_tokens=st.session_state.get("max_tokens", 512),  # 获取生成回复的最大令牌数
                temperature=st.session_state.get("temperature", 0.9),  # 获取生成回复的温度参数
                stream=True,  # 启用流式响应
            ):
                full_response += response.choices[0].delta.content or ""  # 将每次响应的内容拼接到完整响应字符串中

                message_placeholder.markdown(full_response + "▌")  # 显示当前已生成的部分响应，添加一个光标表示正在输入
            message_placeholder.markdown(full_response)  # 显示完整响应

        st.session_state.messages.append(  # 将助手的完整响应添加到消息列表中
=======
                max_tokens=st.session_state.get("max_tokens", 512),
                temperature=st.session_state.get("temperature", 0.9),
                stream=True,
            ):
                full_response += response.choices[0].delta.content or ""

                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            {
                "role": "assistant",
                "content": full_response
            }
        )


<<<<<<< HEAD
if __name__ == "__main__":  # 如果该脚本作为主程序执行
    main()  # 调用主函数

=======
if __name__ == "__main__":
    main()
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
