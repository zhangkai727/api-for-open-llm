from api.config import SETTINGS
from api.models import (
    app,
    EMBEDDING_MODEL,
    LLM_ENGINE,
    RERANK_MODEL,
)


<<<<<<< HEAD
prefix = SETTINGS.api_prefix  # 获取配置中的API前缀

if EMBEDDING_MODEL is not None:
    from api.routes.embedding import embedding_router  # 导入嵌入模型相关的路由

    app.include_router(embedding_router, prefix=prefix, tags=["Embedding"])  # 将嵌入模型路由包含到FastAPI应用中，并添加标签"Embedding"

    try:
        from api.routes.file import file_router  # 尝试导入文件处理相关的路由

        app.include_router(file_router, prefix=prefix, tags=["File"])  # 将文件处理路由包含到FastAPI应用中，并添加标签"File"
    except ImportError:
        pass  # 如果导入失败则忽略

if RERANK_MODEL is not None:
    from api.routes.rerank import rerank_router  # 导入重新排名模型相关的路由

    app.include_router(rerank_router, prefix=prefix, tags=["Rerank"])  # 将重新排名模型路由包含到FastAPI应用中，并添加标签"Rerank"


if LLM_ENGINE is not None:
    from api.routes import model_router  # 导入通用模型相关的路由

    app.include_router(model_router, prefix=prefix, tags=["Model"])  # 将通用模型路由包含到FastAPI应用中，并添加标签"Model"

    if SETTINGS.engine == "vllm":
        from api.vllm_routes import chat_router as chat_router  # 导入VLLM引擎相关的聊天路由
        from api.vllm_routes import completion_router as completion_router  # 导入VLLM引擎相关的完成路由

    elif SETTINGS.engine == "llama.cpp":
        from api.llama_cpp_routes import chat_router as chat_router  # 导入llama.cpp引擎相关的聊天路由
        from api.llama_cpp_routes import completion_router as completion_router  # 导入llama.cpp引擎相关的完成路由

    elif SETTINGS.engine == "tgi":
        from api.tgi_routes import chat_router as chat_router  # 导入TGI引擎相关的聊天路由
        from api.tgi_routes.completion import completion_router as completion_router  # 导入TGI引擎相关的完成路由

    else:
        from api.routes.chat import chat_router as chat_router  # 导入默认引擎相关的聊天路由
        from api.routes.completion import completion_router as completion_router  # 导入默认引擎相关的完成路由

    app.include_router(chat_router, prefix=prefix, tags=["Chat Completion"])  # 将聊天路由包含到FastAPI应用中，并添加标签"Chat Completion"
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])  # 将完成路由包含到FastAPI应用中，并添加标签"Completion"
=======
prefix = SETTINGS.api_prefix

if EMBEDDING_MODEL is not None:
    from api.routes.embedding import embedding_router

    app.include_router(embedding_router, prefix=prefix, tags=["Embedding"])

    try:
        from api.routes.file import file_router

        app.include_router(file_router, prefix=prefix, tags=["File"])
    except ImportError:
        pass

if RERANK_MODEL is not None:
    from api.routes.rerank import rerank_router

    app.include_router(rerank_router, prefix=prefix, tags=["Rerank"])


if LLM_ENGINE is not None:
    from api.routes import model_router

    app.include_router(model_router, prefix=prefix, tags=["Model"])

    if SETTINGS.engine == "vllm":
        from api.vllm_routes import chat_router as chat_router
        from api.vllm_routes import completion_router as completion_router

    else:
        from api.routes.chat import chat_router as chat_router
        from api.routes.completion import completion_router as completion_router

    app.include_router(chat_router, prefix=prefix, tags=["Chat Completion"])
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d


if __name__ == "__main__":
    import uvicorn
<<<<<<< HEAD

    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")  # 运行FastAPI应用，指定主机、端口和日志级别为info

=======
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
