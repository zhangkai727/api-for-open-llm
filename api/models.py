from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import SETTINGS
from api.utils.compat import dictify


def create_app() -> FastAPI:
    import gc  # 导入垃圾回收模块
    import torch  # 导入PyTorch模块

    def torch_gc() -> None:
        r"""
        Collects GPU memory.
        """
        gc.collect()  # 执行垃圾回收
        if torch.cuda.is_available():  # 如果CUDA可用
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA IPC资源

    @asynccontextmanager
    async def lifespan(app: "FastAPI"):  # 生命周期管理器，用于收集GPU内存
        yield  # 返回控制权给调用方
        torch_gc()  # 在生命周期结束时执行GPU内存收集

    """ create fastapi app server """
    app = FastAPI(lifespan=lifespan)  # 创建FastAPI应用程序，指定生命周期管理器
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )  # 添加跨域中间件配置
    return app  # 返回创建的FastAPI应用程序实例


def create_rag_models():
    """ get rag models. """
    rag_models = []  # 初始化RAG模型列表
    if "rag" in SETTINGS.tasks and SETTINGS.activate_inference:  # 如果任务中包含"rag"并且激活推理
        if SETTINGS.embedding_name:  # 如果存在embedding_name设置
            from api.rag import RAGEmbedding  # 导入RAGEmbedding类
            rag_models.append(
               RAGEmbedding(SETTINGS.embedding_name, SETTINGS.embedding_device)  # 创建RAGEmbedding实例并添加到列表中
            )
        else:
            rag_models.append(None)  # 否则添加None到列表中
        if SETTINGS.rerank_name:  # 如果存在rerank_name设置
            from api.rag import RAGReranker  # 导入RAGReranker类
            rag_models.append(
                RAGReranker(SETTINGS.rerank_name, device=SETTINGS.rerank_device)  # 创建RAGReranker实例并添加到列表中
            )
        else:
            rag_models.append(None)  # 否则添加None到列表中
    return rag_models if len(rag_models) == 2 else [None, None]  # 返回RAG模型列表，确保长度为2，不足则填充None


def create_hf_llm():
    """ get generate model for chat or completion. """
    from api.core.default import DefaultEngine  # 导入默认引擎类
    from api.adapter.loader import load_model_and_tokenizer  # 导入加载模型和分词器的函数

    include = {  # 包含在参数字典化中的设置项
        "device_map",
        "load_in_8bit",
        "load_in_4bit",
        "dtype",
        "rope_scaling",
        "flash_attn",
    }
    kwargs = dictify(SETTINGS, include=include)  # 根据设置创建参数字典

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=SETTINGS.model_path, **kwargs,
    )  # 加载模型和分词器

    logger.info("Using default engine")  # 记录使用默认引擎

    return DefaultEngine(
        model,
        tokenizer,
        model_name=SETTINGS.model_name,
        context_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        prompt_name=SETTINGS.chat_template,
    )  # 返回基于默认引擎的生成模型


def create_vllm_engine():
    """ get vllm generate engine for chat or completion. """
    try:
        import vllm  # 尝试导入vllm库
        from vllm.engine.arg_utils import AsyncEngineArgs  # 导入异步引擎参数类
        from vllm.engine.async_llm_engine import AsyncLLMEngine  # 导入异步LLM引擎类
        from api.core.vllm_engine import VllmEngine, LoRA  # 导入VLLM引擎和LoRA类
    except ImportError:
        raise ValueError("VLLM engine not available")  # 若导入失败则抛出错误

    vllm_version = vllm.__version__  # 获取vllm库的版本号

    include = {  # 包含在参数字典化中的设置项
        "tokenizer_mode",
        "trust_remote_code",
        "tensor_parallel_size",
        "dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "enforce_eager",
        "max_loras",
        "max_lora_rank",
        "lora_extra_vocab_size",
        "disable_custom_all_reduce",
    }

    if vllm_version >= "0.4.3":  # 如果vllm版本大于等于0.4.3
        include.add("max_seq_len_to_capture")  # 添加最大序列长度捕获设置项

    kwargs = dictify(SETTINGS, include=include)  # 根据设置创建参数字典
    engine_args = AsyncEngineArgs(
        model=SETTINGS.model_path,
        max_num_batched_tokens=SETTINGS.max_num_batched_tokens if SETTINGS.max_num_batched_tokens > 0 else None,
        max_model_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        quantization=SETTINGS.quantization_method,
        max_cpu_loras=SETTINGS.max_cpu_loras if SETTINGS.max_cpu_loras > 0 else None,
        disable_log_stats=SETTINGS.vllm_disable_log_stats,
        disable_log_requests=True,
        **kwargs,
    )  # 创建异步引擎参数实例

    engine = AsyncLLMEngine.from_engine_args(engine_args)  # 基于引擎参数创建异步LLM引擎实例

    logger.info("Using vllm engine")  # 记录使用vllm引擎

    lora_modules = []
    for item in SETTINGS.lora_modules.strip().split("+"):
        if "=" in item:
            name, path = item.split("=")
            lora_modules.append(LoRA(name, path))  # 根据设置创建LoRA模块列表

    return VllmEngine(
        engine,
        SETTINGS.model_name,
        SETTINGS.chat_template,
        lora_modules=lora_modules,
    )  # 返回基于VLLM引擎的生成引擎实例



def create_llama_cpp_engine():
    """ get llama.cpp generate engine for chat or completion. """
    try:
        from llama_cpp import Llama  # 尝试导入llama.cpp的Llama类
        from api.core.llama_cpp_engine import LlamaCppEngine  # 导入llama.cpp引擎类
    except ImportError:
        raise ValueError("Llama cpp engine not available")  # 若导入失败则抛出错误

    include = {  # 包含在参数字典化中的设置项
        "n_gpu_layers",
        "main_gpu",
        "tensor_split",
        "n_batch",
        "n_threads",
        "n_threads_batch",
        "rope_scaling_type",
        "rope_freq_base",
        "rope_freq_scale",
    }
    kwargs = dictify(SETTINGS, include=include)  # 根据设置创建参数字典
    engine = Llama(
        model_path=SETTINGS.model_path,
        n_ctx=SETTINGS.context_length if SETTINGS.context_length > 0 else 2048,
        **kwargs,
    )  # 创建llama.cpp引擎实例

    logger.info("Using llama.cpp engine")  # 记录使用llama.cpp引擎

    return LlamaCppEngine(engine, SETTINGS.model_name, SETTINGS.chat_template)  # 返回基于llama.cpp引擎的生成引擎实例


def create_tgi_engine():
    """ get tgi generate engine for chat or completion. """
    try:
        from text_generation import AsyncClient  # 尝试导入文本生成异步客户端
        from api.core.tgi import TGIEngine  # 导入TGI引擎类
    except ImportError:
        raise ValueError("TGI engine not available")  # 若导入失败则抛出错误

    client = AsyncClient(SETTINGS.tgi_endpoint)  # 创建文本生成异步客户端实例
    logger.info("Using TGI engine")  # 记录使用TGI引擎

    return TGIEngine(client, SETTINGS.model_name, SETTINGS.chat_template)  # 返回基于TGI引擎的生成引擎实例


# fastapi app
app = create_app()  # 创建FastAPI应用实例

# model for rag
EMBEDDING_MODEL, RERANK_MODEL = create_rag_models()  # 获取RAG模型

# llm
if "llm" in SETTINGS.tasks and SETTINGS.activate_inference:
    if SETTINGS.engine == "default":
        LLM_ENGINE = create_hf_llm()  # 创建基于默认引擎的LLM引擎
    elif SETTINGS.engine == "vllm":
        LLM_ENGINE = create_vllm_engine()  # 创建VLLM引擎
    elif SETTINGS.engine == "llama.cpp":
        LLM_ENGINE = create_llama_cpp_engine()  # 创建llama.cpp引擎
    elif SETTINGS.engine == "tgi":
        LLM_ENGINE = create_tgi_engine()  # 创建TGI引擎
else:
    LLM_ENGINE = None  # 如果没有LLM任务或未激活推断，则LLM引擎为空

# model names for special processing
EXCLUDE_MODELS = ["baichuan-13b", "baichuan2-13b", "qwen", "chatglm3"]  # 特殊处理的模型名称列表

