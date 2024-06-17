from abc import ABC, abstractmethod
from typing import (
    List,
    Dict,
    Any,
    Optional,
)

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from api.utils.protocol import (
    DocumentObj,
    Document,
    RerankResponse,
)


class BaseReranker(ABC):  # 定义一个抽象基类 BaseReranker
    @abstractmethod  # 定义抽象方法的装饰器，表示该方法在子类中必须实现
    @torch.inference_mode()  # 禁用梯度计算，以节省内存和提高推理速度
    def rerank(  # 定义一个抽象方法 rerank
        self,  # 方法的第一个参数总是 self
        query: str,  # 输入参数 query，类型为字符串，表示查询
        documents: List[str],  # 输入参数 documents，类型为字符串列表，表示文档列表
        batch_size: Optional[int] = 256,  # 输入参数 batch_size，可选整数，表示批处理大小，默认值为 256
        top_n: Optional[int] = None,  # 输入参数 top_n，可选整数，表示返回的前 n 个结果
        return_documents: Optional[bool] = False,  # 输入参数 return_documents，可选布尔值，表示是否返回文档，默认值为 False
    ) -> Dict[str, Any]:  # 方法返回值的类型为字典，键和值的类型为任意类型
        ...  # 方法体省略，表示这是一个抽象方法，需要在子类中实现


class RAGReranker(BaseReranker):  # 定义一个类 RAGReranker，继承自 BaseReranker
    def __init__(  # 定义类的初始化方法
        self,  # 方法的第一个参数总是 self
        model_name_or_path: str,  # 输入参数 model_name_or_path，类型为字符串，表示模型名称或路径
        device: str = None,  # 输入参数 device，类型为字符串，表示设备，默认值为 None
    ) -> None:  # 方法没有返回值
        self.client = CrossEncoder(  # 创建一个 CrossEncoder 实例，并赋值给实例属性 client
            model_name_or_path,  # 使用模型名称或路径
            device=device,  # 使用指定的设备
            trust_remote_code=True,  # 信任远程代码
        )
        logger.info(f"Loading from `{model_name_or_path}`.")  # 记录日志，输出模型加载信息

    @torch.inference_mode()  # 禁用梯度计算，以节省内存和提高推理速度
    def rerank(  # 定义 rerank 方法，实现文档重排序
        self,  # 方法的第一个参数总是 self
        query: str,  # 输入参数 query，类型为字符串，表示查询
        documents: List[str],  # 输入参数 documents，类型为字符串列表，表示文档列表
        batch_size: Optional[int] = 256,  # 输入参数 batch_size，可选整数，表示批处理大小，默认值为 256
        top_n: Optional[int] = None,  # 输入参数 top_n，可选整数，表示返回的前 n 个结果
        return_documents: Optional[bool] = False,  # 输入参数 return_documents，可选布尔值，表示是否返回文档，默认值为 False
        **kwargs: Any,  # 其他可选参数
    ) -> Optional[RerankResponse]:  # 方法返回值的类型为可选的 RerankResponse
        results = self.client.rank(  # 调用 client 的 rank 方法，进行文档重排序
            query=query,  # 传入查询
            documents=documents,  # 传入文档列表
            top_k=top_n,  # 传入前 n 个结果
            return_documents=True,  # 指定返回文档
            batch_size=batch_size,  # 指定批处理大小
            **kwargs,  # 传入其他可选参数
        )

        if return_documents:  # 如果需要返回文档
            docs = [  # 创建包含文档信息的列表
                DocumentObj(  # 创建 DocumentObj 实例
                    index=int(res["corpus_id"]),  # 文档索引
                    relevance_score=float(res["score"]),  # 相关性得分
                    document=Document(text=res["text"]),  # 文档内容
                )
                for res in results  # 遍历结果
            ]
        else:  # 如果不需要返回文档
            docs = [  # 创建不包含文档信息的列表
                DocumentObj(  # 创建 DocumentObj 实例
                    index=int(res["corpus_id"]),  # 文档索引
                    relevance_score=float(res["score"]),  # 相关性得分
                    document=None,  # 文档内容为空
                )
                for res in results  # 遍历结果
            ]
        return RerankResponse(results=docs)  # 返回包含重排序结果的 RerankResponse 实例

