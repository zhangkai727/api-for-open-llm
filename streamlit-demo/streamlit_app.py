import os

import streamlit as st

from streamlit_gallery.utils.page import page_group


def main():
    # 导入所需的模块和函数
    from streamlit_gallery.apps import gallery
    from streamlit_gallery.components import chat, doc_chat
    # 导入自定义的页面组管理函数
    page = page_group("p")

    # 在 Streamlit 的侧边栏中设置标题和展开/折叠的部件
    with st.sidebar:
        st.title("🎉 LLM Gallery")

        # 展开 'APPS' 部分，并设置默认选项为展示 LLM Chat Gallery
        with st.expander("✨ APPS", True):
            page.item("LLM Chat Gallery", gallery, default=True)

        # 展开 'COMPONENTS' 部分，并根据环境变量的配置动态展示不同的组件选项
        with st.expander("🧩 COMPONENTS", True):
            # 如果存在 CHAT_API_BASE 环境变量，则展示 'Chat' 和 'Doc Chat' 组件选项
            if os.getenv("CHAT_API_BASE", ""):
                page.item("Chat", chat)
                page.item("Doc Chat", doc_chat)

            # 如果存在 SQL_CHAT_API_BASE 环境变量，则动态导入并展示 'SQL Chat' 组件选项
            if os.getenv("SQL_CHAT_API_BASE", ""):
                from streamlit_gallery.components import sql_chat
                page.item("SQL Chat", sql_chat)

            # 如果存在 SERPAPI_API_KEY 环境变量，则动态导入并展示 'Search Chat' 组件选项
            if os.getenv("SERPAPI_API_KEY", ""):
                from streamlit_gallery.components import search_chat
                page.item("Search Chat", search_chat)

            # 如果存在 TOOL_CHAT_API_BASE 环境变量，则动态导入并展示 'Tool Chat' 组件选项
            if os.getenv("TOOL_CHAT_API_BASE", ""):
                from streamlit_gallery.components import tool_chat
                page.item("Tool Chat", tool_chat)

            # 如果存在 INTERPRETER_CHAT_API_BASE 环境变量，则动态导入并展示 'Code Interpreter' 组件选项
            if os.getenv("INTERPRETER_CHAT_API_BASE", ""):
                from streamlit_gallery.components import code_interpreter
                page.item("Code Interpreter", code_interpreter)

        # 添加一个按钮，用于清空会话中的消息记录
        if st.button("🗑️ 清空消息"):
            st.session_state.messages = []

        # 展开 '模型配置' 部分，并设置模型名称、接口地址和 API KEY 的输入框
        with st.expander("✨ 模型配置", False):
            model_name = st.text_input(label="模型名称")
            base_url = st.text_input(label="模型接口地址", value=os.getenv("CHAT_API_BASE"))
            api_key = st.text_input(label="API KEY", value=os.getenv("API_KEY", "xxx"))

            # 更新会话状态中的模型配置信息
            st.session_state.update(
                dict(
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
            )

        # 展开 '参数配置' 部分，并设置滑动条用于调整不同的参数
        with st.expander("🐧 参数配置", False):
            max_tokens = st.slider("回复最大token数量", 20, 4096, 1024)
            temperature = st.slider("温度", 0.0, 1.0, 0.9)
            chunk_size = st.slider("文档分块大小", 100, 512, 250)
            chunk_overlap = st.slider("文档分块重复大小", 0, 100, 50)
            top_k = st.slider("文档分块检索数量", 0, 10, 4)

            # 更新会话状态中的参数配置信息
            st.session_state.update(
                dict(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )
            )

    # 展示当前选择的页面组件
    page.show()

if __name__ == "__main__":
    # 设置页面配置，包括标题、图标和布局
    st.set_page_config(page_title="Streamlit LLM Gallery", page_icon="🎈", layout="wide")
    main()
