from typing import List, Optional

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

answer_prompt = PromptTemplate.from_template(
    """给出以下用户问题、相应的 SQL 查询和 SQL 结果，请回答用户问题。

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)


def create_sql_query(
    query: str,  # 用户提供的查询问题
    base_url: str,  # SQL Chatbot 的基础 URL
    database_uri: str,  # 数据库连接 URI
    include_tables: Optional[List[str]] = None,  # 要包括在查询中的表格名称列表（可选）
    sample_rows_in_table_info: Optional[int] = 1,  # 在获取表格信息时采样的行数（可选，默认为 1）
):
    question = {"question": query}  # 构建问题字典

    # 从数据库连接 URI 创建 SQLDatabase 对象
    db = SQLDatabase.from_uri(
        database_uri,
        include_tables=include_tables,
        sample_rows_in_table_info=sample_rows_in_table_info,
    )

    # 创建 ChatOpenAI 对象，用于执行自然语言处理任务
    llm = ChatOpenAI(
        model="codeqwen",
        temperature=0,
        openai_api_base=base_url,
        openai_api_key="xxx"
    )

    # 创建执行 SQL 查询语句的链式操作对象
    write = create_sql_query_chain(llm, db)

    # 调用链式操作对象生成 SQL 查询语句并执行查询
    sql_query = write.invoke(question)
    sql_result = db.run(sql_query, fetch="cursor")

    return sql_query, sql_result


def create_llm_chain(base_url: str):
    # 创建 ChatOpenAI 对象，用于执行自然语言处理任务
    llm = ChatOpenAI(
        model="codeqwen",
        temperature=0,
        openai_api_base=base_url,
        openai_api_key="xxx"
    )

    # 返回语言模型链对象，包括问题回答、语言模型和字符串输出解析器
    return answer_prompt | llm | StrOutputParser()


if __name__ == "__main__":
    import pandas as pd

    # 示例代码，调用 create_sql_query 函数执行查询
    sql_query, sql_result = create_sql_query(
        "2024年各个信息来源分别发布了多少资讯,按照数量排序",
        base_url="http://192.168.20.44:7861/v1",
        include_tables=["document", "source"],
        database_uri="mysql+pymysql://root:Dnect_123@192.168.0.52:3306/information_services",
    )
    print(pd.DataFrame(sql_result))  # 打印查询结果的 DataFrame
