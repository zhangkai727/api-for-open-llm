from fastapi import APIRouter, Depends, status

from api.models import RERANK_MODEL
from api.rag import RAGReranker
from api.utils.protocol import (
    RerankRequest,
)
from api.utils.request import check_api_key

rerank_router = APIRouter()


def get_embedding_engine():
    yield RERANK_MODEL  # 返回重新排序模型引擎对象 RERANK_MODEL


@rerank_router.post(
    "/rerank",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def create_rerank(request: RerankRequest, client: RAGReranker = Depends(get_embedding_engine)):

    return client.rerank(
        query=request.query,  # 查询文本
        documents=request.documents,  # 待重新排序的文档列表
        top_n=request.top_n,  # 返回前n个最相关的文档
        return_documents=request.return_documents,  # 是否返回重新排序后的文档列表
    )

