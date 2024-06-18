import tiktoken
from fastapi import APIRouter, Depends, status

from api.config import SETTINGS
from api.models import EMBEDDING_MODEL
<<<<<<< HEAD
from api.rag import RAGEmbedding
from api.utils.protocol import EmbeddingCreateParams
from api.utils.request import check_api_key
=======
from api.protocol import EmbeddingCreateParams
from api.rag import RAGEmbedding
from api.utils import check_api_key
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d

embedding_router = APIRouter()


def get_embedding_engine():
    yield EMBEDDING_MODEL

<<<<<<< HEAD
# 定义了一个生成器函数，用于异步生成嵌入模型引擎实例
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d

@embedding_router.post(
    "/embeddings",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
@embedding_router.post(
    "/engines/{model_name}/embeddings",
)
async def create_embeddings(
    request: EmbeddingCreateParams,
    model_name: str = None,
    client: RAGEmbedding = Depends(get_embedding_engine),
):
<<<<<<< HEAD
    """为文本创建嵌入"""
    # 如果请求中未提供模型名称，则使用路径参数中的model_name
    if request.model is None:
        request.model = model_name

    # 处理输入文本，确保它以列表形式提供给嵌入模型
=======
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    request.input = request.input
    if isinstance(request.input, str):
        request.input = [request.input]
    elif isinstance(request.input, list):
        if isinstance(request.input[0], int):
            decoding = tiktoken.model.encoding_for_model(request.model)
            request.input = [decoding.decode(request.input)]
        elif isinstance(request.input[0], list):
            decoding = tiktoken.model.encoding_for_model(request.model)
            request.input = [decoding.decode(text) for text in request.input]

<<<<<<< HEAD
    # 设置嵌入维度，默认从SETTINGS中获取，如果未设置，则为-1
    request.dimensions = request.dimensions or getattr(SETTINGS, "embedding_size", -1)

    # 调用嵌入模型客户端的embed方法生成嵌入
=======
    request.dimensions = request.dimensions or getattr(SETTINGS, "embedding_size", -1)

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    return client.embed(
        texts=request.input,
        model=request.model,
        encoding_format=request.encoding_format,
        dimensions=request.dimensions,
    )
<<<<<<< HEAD

=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
