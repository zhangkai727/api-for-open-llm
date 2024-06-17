import os
import secrets
import uuid
from pathlib import Path
from typing import List, Optional

import lancedb
import pyarrow as pa
import requests
from lancedb.rerankers import CohereReranker
from loguru import logger
from openai import OpenAI

# 获取环境变量 EMBEDDING_API_BASE 和 CO_API_URL
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE")
CO_API_URL = os.getenv("CO_API_URL")

# 如果没有设置 CO_API_URL，则将 EMBEDDING_API_BASE 赋值给 CO_API_URL
if not CO_API_URL:
    os.environ["CO_API_URL"] = EMBEDDING_API_BASE

# 定义一个自定义的重排序类 RefinedCohereReranker 继承自 CohereReranker
class RefinedCohereReranker(CohereReranker):
    def _rerank(self, result_set: pa.Table, query: str):
        # 提取文档列表
        docs = result_set[self.column].to_pylist()
        # 使用 Cohere API 进行重排序
        results = self._client.rerank(
            query=query,
            documents=docs,
            top_n=self.top_n,
            model=self.model_name,
        )
        # 获取重排序后的索引和分数
        indices, scores = list(
            zip(*[(result.index, result.relevance_score) for result in results.results])
        )
        # 根据重排序结果重新排列 result_set
        result_set = result_set.take(list(indices))
        # 添加重排序分数列
        result_set = result_set.append_column(
            "_relevance_score", pa.array(scores, type=pa.float32())
        )

        return result_set

# 定义一个文档服务器类 DocServer
class DocServer:
    def __init__(self, embeddings):
        # 初始化嵌入对象
        self.embeddings = embeddings
        self.vector_store = None
        self.vs_path = None
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=os.getenv("EMBEDDING_API_BASE"),
            api_key=os.getenv("API_KEY", ""),
        )
        # 连接到 LanceDB 数据库
        self.db = lancedb.connect(
            os.path.join(Path(__file__).parents[3], "lancedb.db"),
        )

    # 上传文件或 URL，并创建索引
    def upload(
        self,
        filepath: Optional[str] = None,
        url: Optional[str] = None,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        table_name: str = None,
    ) -> str:
        # 如果上传的是 URL
        if url is not None:
            data = {
                "url": url,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
            file_id = str(secrets.token_hex(12))
        else:
            # 上传文件到 OpenAI 并获取文件 ID 和文件名
            upf = self.client.files.create(file=open(filepath, "rb"), purpose="assistants")
            file_id, filename = upf.id, upf.filename
            data = {
                "file_id": file_id,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

        # 发送请求到嵌入 API 以分割文件
        res = requests.post(
            url=os.getenv("EMBEDDING_API_BASE") + "/files/split", json=data,
        ).json()

        # 如果没有提供表名，则使用文件 ID 作为表名
        table_name = table_name or file_id
        # 嵌入文档
        embeddings = self.embeddings.embed_documents(
            [doc["page_content"] for doc in res["docs"]]
        )
        data = []
        for i, doc in enumerate(res["docs"]):
            # 准备要插入的数据
            append_data = {
                "id": str(uuid.uuid4()),
                "vector": embeddings[i],
                "text": doc["page_content"],
                "metadata": doc["metadata"]["source"],
            }
            data.append(append_data)

        # 将数据插入到数据库中
        if table_name in self.db.table_names():
            tbl = self.db.open_table(table_name)
            tbl.add(data)
        else:
            self.db.create_table(table_name, data)

        # 打印插入成功的信息
        logger.info("Successfully inserted documents!")

        return table_name

    # 搜索方法
    def search(
        self,
        query: str,
        top_k: int,
        table_name: str,
        rerank: bool = False,
    ) -> List:
        # 打开表
        table = self.db.open_table(table_name)
        # 嵌入查询
        embedding = self.embeddings.embed_query(query)
        top_n = 2 * top_k if rerank else top_k

        # 搜索文档
        docs = table.search(
            embedding,
            vector_column_name="vector",
        ).metric("cosine").limit(top_n)

        # 如果需要重排序
        if rerank:
            docs = docs.rerank(
                reranker=RefinedCohereReranker(api_key="xxx"),
                query_string=query,
            ).limit(top_k)

        # 转换为 Pandas DataFrame
        docs = docs.to_pandas()
        # 删除向量列和 ID 列
        del docs["vector"]
        del docs["id"]
        return docs

    # 删除索引方法
    def delete(self, table_name):
        if table_name:
            self.db.drop_table(table_name)
            try:
                # 尝试删除文件
                self.client.files.delete(file_id=table_name)
            except:
                pass
        return table_name

# 定义文档问答的提示模板
DOCQA_PROMPT = """参考信息：
{context}
---
我的问题或指令：
{query}
---
请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
你的回复："""

