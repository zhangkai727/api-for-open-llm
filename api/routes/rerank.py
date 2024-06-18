from fastapi import APIRouter, Depends, status

from api.models import RERANK_MODEL
<<<<<<< HEAD
from api.rag import RAGReranker
from api.utils.protocol import (
    RerankRequest,
)
from api.utils.request import check_api_key
=======
from api.protocol import RerankRequest
from api.rag import RAGReranker
from api.utils import check_api_key
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d

rerank_router = APIRouter()


def get_embedding_engine():
<<<<<<< HEAD
    yield RERANK_MODEL  # 返回重新排序模型引擎对象 RERANK_MODEL
=======
    yield RERANK_MODEL
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d


@rerank_router.post(
    "/rerank",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def create_rerank(request: RerankRequest, client: RAGReranker = Depends(get_embedding_engine)):
<<<<<<< HEAD

    return client.rerank(
        query=request.query,  # 查询文本
        documents=request.documents,  # 待重新排序的文档列表
        top_n=request.top_n,  # 返回前n个最相关的文档
        return_documents=request.return_documents,  # 是否返回重新排序后的文档列表
    )

=======
    return client.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
        return_documents=request.return_documents,
    )
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
