import base64
import os
from abc import ABC
from typing import (
    List,
    Literal,
    Optional,
    Sequence,
)

import numpy as np
from openai.types.create_embedding_response import Usage
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings

from api.utils.protocol import CreateEmbeddingResponse, Embedding


class BaseEmbedding(ABC):  # 定义一个抽象基类 BaseEmbedding
    def embed(  # 定义一个嵌入方法 embed
        self,  # 方法的第一个参数总是 self
        texts: Sequence[str],  # 输入参数 texts，是一个字符串序列
        model: Optional[str] = "bce",  # 输入参数 model，可选字符串，默认值为 "bce"
        encoding_format: Literal["float", "base64"] = "float",  # 输入参数 encoding_format，支持 "float" 或 "base64"，默认值为 "float"
    ) -> CreateEmbeddingResponse:  # 方法返回值的类型为 CreateEmbeddingResponse
        ...

class RAGEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = None,
    ) -> None:
        self.client = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=True,
        )

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = "bce",
        encoding_format: Literal["float", "base64"] = "float",
        dimensions: Optional[int] = -1,
    ) -> CreateEmbeddingResponse:
        dim = self.client.get_sentence_embedding_dimension()  # 获取句子嵌入的维度
        use_matryoshka = bool(0 < dimensions < dim)  # 检查是否需要使用 Matryoshka

        data, total_tokens = [], 0  # 初始化数据和总 token 数量
        batches = [texts[i: i + 1024] for i in range(0, len(texts), 1024)]  # 将文本列表分批处理，每批最多1024个文本

        for num_batch, batch in enumerate(batches):  # 遍历每一个批次
            vecs = self.client.encode(
                batch,  # 当前批次的文本
                batch_size=int(os.getenv("batch_size", 32)),  # 每个批次的大小，从环境变量中获取，默认为32
                normalize_embeddings=False if use_matryoshka else True,  # 是否规范化嵌入向量
                convert_to_tensor=True if use_matryoshka else False,  # 是否转换为张量
            )

            bs = vecs.shape[0]  # 当前批次的大小
            if dimensions > dim:
                zeros = np.zeros((bs, dimensions - dim))  # 创建一个零向量用于填充
                vecs = np.c_[vecs, zeros]  # 将零向量拼接到嵌入向量后面
            elif 0 < dimensions < dim:
                vecs = vecs[..., :dimensions]  # 缩减嵌入向量的维度
                vecs = normalize_embeddings(vecs).cpu().numpy()  # 规范化嵌入向量并转换为NumPy数组

            if encoding_format == "base64":
                vecs = [base64.b64encode(v.tobytes()).decode("utf-8") for v in vecs]  # 将嵌入向量转换为base64格式
            else:
                vecs = vecs.tolist()  # 将嵌入向量转换为列表格式

            data.extend(
                Embedding(
                    index=num_batch * 1024 + i,  # 当前嵌入的索引
                    object="embedding",  # 嵌入对象类型
                    embedding=embedding,  # 嵌入向量
                )
                for i, embedding in enumerate(vecs)  # 遍历嵌入向量并创建嵌入对象
            )
            total_tokens += sum(len(i) for i in batch)  # 更新总的 token 数量

        return CreateEmbeddingResponse(
            data=data,  # 嵌入数据列表
            model=model,  # 使用的模型
            object="list",  # 返回对象类型为列表
            usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens),  # 使用信息
        )
