import os
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from .utils import DocServer, DOCQA_PROMPT


def main():
<<<<<<< HEAD
    # 设置上传文件夹路径，位于脚本文件的上三级目录中的 "upload_files" 文件夹
    UPLOAD_FOLDER = os.path.join(Path(__file__).parents[3], "upload_files")

    # 创建上传文件夹（如果不存在）
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # 缓存加载文档服务器
    @st.cache_resource
    def load_doc_server():
        # 初始化 OpenAI Embeddings
=======
    UPLOAD_FOLDER = os.path.join(Path(__file__).parents[3], "upload_files")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    @st.cache_resource
    def load_doc_server():
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        embeddings = OpenAIEmbeddings(
            openai_api_base=os.getenv("EMBEDDING_API_BASE"),
            openai_api_key=os.getenv("API_KEY", ""),
        )
<<<<<<< HEAD
        # 创建文档服务器实例
        server = DocServer(embeddings)
        return server

    # 加载文档服务器
    server = load_doc_server()

    # 缓存创建文件索引
    @st.cache_resource
    def create_file_index(file, chunk_size, chunk_overlap, table_name):
        # 获取文件名并设置文件路径
        filename = file.name
        filepath = f"{UPLOAD_FOLDER}/{filename}"
        # 保存上传的文件
        with open(filepath, "wb") as f:
            f.write(file.read())

        # 上传文件并创建索引，获取文件ID
=======
        server = DocServer(embeddings)
        return server

    server = load_doc_server()

    @st.cache_resource
    def create_file_index(file, chunk_size, chunk_overlap, table_name):
        filename = file.name
        filepath = f"{UPLOAD_FOLDER}/{filename}"
        with open(filepath, "wb") as f:
            f.write(file.read())

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        file_id = server.upload(
            filepath,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            table_name=table_name,
        )
<<<<<<< HEAD
        # 更新会话状态
        st.session_state.update(dict(file_id=file_id))

        # 删除本地保存的文件
        os.remove(filepath)
        return file.name

    # 缓存创建URL索引
    @st.cache_resource
    def create_url_index(url, chunk_size, chunk_overlap, table_name):
        # 上传URL并创建索引，获取表名
=======
        st.session_state.update(dict(file_id=file_id))

        os.remove(filepath)
        return file.name

    @st.cache_resource
    def create_url_index(url, chunk_size, chunk_overlap, table_name):
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        table_name = server.upload(
            url=url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            table_name=table_name,
        )
        return table_name

<<<<<<< HEAD
    # 删除索引
    def delete_index(table_name):
        # 删除指定表名的索引
        server.delete(table_name)
        return table_name

    # 设置标题
    st.title("💬 Document Chatbot")

    # 初始化OpenAI客户端
=======
    def delete_index(table_name):
        server.delete(table_name)
        return table_name

    st.title("💬 Document Chatbot")

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    client = OpenAI(
        api_key=st.session_state.get("api_key", "xxx"),
        base_url=st.session_state.get("base_url", "xxx"),
    )

<<<<<<< HEAD
    # 设置页面布局
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    col1, col2, col3 = st.columns([3, 3, 4])

    with col1:
        with st.expander(label="✨ 简介"):
<<<<<<< HEAD
            # 显示简介
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            st.markdown("""+ 文档问答是指从文本或文档中检索和理解相关信息，然后回答用户提出的问题。

+ 该技术通常用于信息检索、知识图谱问答、智能客服等领域。

+ 本项目支持**文档问答**和**URL问答**""")
<<<<<<< HEAD
            # 选择上传文件类型
            mode = st.selectbox("请选择上传文件类型", options=["文件", "网址"])
            # 选择是否重排序
=======
            mode = st.selectbox("请选择上传文件类型", options=["文件", "网址"])
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            rerank = st.checkbox("🚀 重排序")

    with col2:
        with st.expander("📖 知识库列表", False):
<<<<<<< HEAD
            # 显示知识库列表
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            vector_store_names = server.db.table_names()
            st.dataframe(pd.DataFrame({"vector_store_name": vector_store_names}))

    with col3:
        with st.expander("📚‍ 配置"):
<<<<<<< HEAD
            # 输入网址
            url = st.text_input("网址", placeholder="https://qwenlm.github.io/zh/blog/codeqwen1.5/")
            # 上传文件
            file = st.file_uploader("上传文件", accept_multiple_files=False)

            # 选择或创建知识库
=======
            url = st.text_input("网址", placeholder="https://qwenlm.github.io/zh/blog/codeqwen1.5/")
            file = st.file_uploader("上传文件", accept_multiple_files=False)

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            table_name = st.text_input(
                "选择或者创建知识库",
                placeholder=vector_store_names[0] if vector_store_names else "test"
            )

            col5, col6 = st.columns([5, 5])
            with col5:
                create = st.button("✅ 导入知识库")
            with col6:
                if st.button("❌ 删除知识库"):
                    _ = delete_index(table_name)

<<<<<<< HEAD
            # 根据文件和模式创建文件索引
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            if file and mode == "文件" and table_name and create:
                create_file_index(
                    file,
                    chunk_size=st.session_state.get("chunk_size", 250),
                    chunk_overlap=st.session_state.get("chunk_overlap", 50),
                    table_name=table_name,
                )

<<<<<<< HEAD
            # 根据URL和模式创建URL索引
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            if url and mode == "网址" and table_name and create:
                create_url_index(
                    url,
                    chunk_size=st.session_state.get("chunk_size", 250),
                    chunk_overlap=st.session_state.get("chunk_overlap", 50),
                    table_name=table_name,
                )

<<<<<<< HEAD
            # 更新会话状态
            st.session_state.update(dict(table_name=table_name))

    # 初始化消息列表
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示会话中的消息
=======
            st.session_state.update(dict(table_name=table_name))

    if "messages" not in st.session_state:
        st.session_state.messages = []

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if message["role"] == "assistant" and isinstance(message["reference"], pd.DataFrame):
            with st.expander(label="展示搜索结果"):
                st.dataframe(message["reference"], use_container_width=True)

<<<<<<< HEAD
    # 处理用户输入的消息
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    if prompt := st.chat_input("What is up?"):
        table_name = st.session_state.get("table_name", None)
        doc_prompt, reference = None, None
        if table_name is not None:
<<<<<<< HEAD
            # 在知识库中搜索相关信息
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            result = server.search(
                query=prompt,
                top_k=st.session_state.get("top_k", 3),
                table_name=table_name,
                rerank=rerank,
            )

            context = "\n\n".join(doc for doc in result["text"].tolist())
            doc_prompt = DOCQA_PROMPT.format(query=prompt, context=context)
            reference = result

<<<<<<< HEAD
        # 更新会话状态
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            pyload = dict(
                model=st.session_state.get("model_name", "xxx"),
                messages=[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages[:-1]
                ] + [
                        {
                            "role": "user",
                            "content": doc_prompt or prompt
                        }
                ],
                stream=True,
                max_tokens=st.session_state.get("max_tokens", 512),
                temperature=st.session_state.get("temperature", 0.9),
            )

<<<<<<< HEAD
            # 生成模型回复
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            for response in client.chat.completions.create(**pyload):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            if isinstance(reference, pd.DataFrame):
                with st.expander(label="展示搜索结果"):
                    st.dataframe(reference, use_container_width=True)

<<<<<<< HEAD
        # 更新会话状态
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "reference": reference,
            }
        )


if __name__ == "__main__":
<<<<<<< HEAD
    main()  # 运行主函数

=======
    main()
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
