import os

import pandas as pd
import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase

from .utils import create_sql_query, create_llm_chain


import os
import pandas as pd
import streamlit as st
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import create_engine, Table, MetaData, select

def main():
    st.title("💬 SQL Chatbot")  # 设置页面标题

    base_url = os.getenv("SQL_CHAT_API_BASE")  # 从环境变量中获取 SQL 服务的基本 URL
    col1, col2 = st.columns(2)  # 创建页面布局，分为两列

    with col1:
        with st.expander(label="✨ 简介"):  # 展开框部分，显示 SQL 问答流程的简介信息
            st.markdown("""+ SQL问答流程：
    + 基于用户问题和选定表结构生成可执行的 sql 语句
    + 执行 sql 语句，返回数据库查询结果
    + [TODO] 通过 schema link 自动寻找相关的表
    + [TODO] 根据查询结果对用户问题进行回复""")

    with col2:
        with st.expander("🐬 数据库配置", False):  # 展开框部分，显示数据库配置
            db_url = st.text_input("URL", placeholder="mysql+pymysql://")  # 输入框，获取数据库连接 URL
            if db_url:
                try:
                    db = SQLDatabase.from_uri(database_uri=db_url)  # 使用数据库连接 URL 创建 SQLDatabase 对象
                    table_names = db.get_usable_table_names()  # 获取可用表格名称列表
                except:
                    table_names = []  # 若连接失败，则表格列表为空
                    st.error("Wrong configuration for database connection!")  # 显示错误信息，指示数据库连接配置错误

                include_tables = st.multiselect("选择查询表", table_names)  # 多选框，选择要查询的表格

    if "messages" not in st.session_state:
        st.session_state.messages = []  # 如果会话状态中没有 messages 键，则初始化为空列表

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])  # 如果角色是用户，则显示用户的输入内容
            else:
                st.markdown(message["content"])  # 如果角色是助手，则显示助手的回复内容
                st.markdown("### SQL Query")
                if message["sql"] is not None:
                    st.code(message["sql"], language="sql")  # 如果存在 SQL 查询语句，则显示 SQL 代码
                if message["data"] is not None:
                    with st.expander("展示查询结果"):
                        st.dataframe(message["data"], use_container_width=True)  # 如果存在查询结果，则以表格形式展示

    if query := st.chat_input("2022年xx大学参与了哪些项目？"):  # 获取用户的查询输入
        st.session_state.messages.append({"role": "user", "content": query})  # 将用户输入添加到会话状态的消息列表中
        with st.chat_message("user"):
            st.markdown(query)  # 显示用户输入的查询内容

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            sql_query, sql_result = create_sql_query(query, base_url, db_url, include_tables)  # 创建 SQL 查询语句和查询结果
            data = pd.DataFrame(sql_result) if sql_result else None  # 将查询结果转换为 DataFrame，如果没有结果则为 None
            str_data = data.to_markdown() if data is not None else ""  # 将 DataFrame 转换为 Markdown 格式的字符串

            llm_chain = create_llm_chain(base_url)  # 创建语言模型链
            for chunk in llm_chain.stream(
                {"question": query, "query": sql_query, "result": str_data}
            ):
                full_response += chunk or ""  # 将每个响应片段添加到完整的响应中
                message_placeholder.markdown(full_response + "▌")  # 在页面上显示完整的响应

            message_placeholder.markdown(full_response)  # 在页面上显示完整的响应
            if sql_query:
                st.markdown("### SQL Query")
                st.code(sql_query, language="sql")  # 如果存在 SQL 查询语句，则显示 SQL 代码

            if data is not None:
                with st.expander("展示查询结果"):
                    st.dataframe(data, use_container_width=True)  # 如果存在查询结果，则以表格形式展示

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sql": sql_query,
                "data": data,
            }
        )


if __name__ == "__main__":
    main()  # 调用主函数，启动 SQL Chatbot 的交互界面

