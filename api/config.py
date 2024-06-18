<<<<<<< HEAD
import multiprocessing
=======
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
import os
from pathlib import Path
from typing import Optional, Dict, List, Union

import dotenv
from loguru import logger
from pydantic import BaseModel, Field

<<<<<<< HEAD
from api.utils.compat import jsonify, disable_warnings

dotenv.load_dotenv()  # 加载环境变量文件中的配置

disable_warnings(BaseModel)  # 禁用BaseModel的警告信息

def get_bool_env(key, default="false"):
    return os.environ.get(key, default).lower() == "true"  # 获取布尔类型的环境变量值

def get_env(key, default):
    val = os.environ.get(key, "")  # 获取环境变量的值
    return val or default  # 返回环境变量的值或默认值

ENGINE = get_env("ENGINE", "default").lower()  # 获取并转换小写ENGINE配置
TEI_ENDPOINT = get_env("TEI_ENDPOINT", None)  # 获取TEI_ENDPOINT配置
TASKS = get_env("TASKS", "llm").lower().split(",")  # 获取并转换小写TASKS配置，并按逗号分割为列表

# 获取STORAGE_LOCAL_PATH配置，如果不存在则设置默认路径并创建目录
=======
from api.common import jsonify, disable_warnings

dotenv.load_dotenv()

disable_warnings(BaseModel)


def get_bool_env(key, default="false"):
    return os.environ.get(key, default).lower() == "true"


def get_env(key, default):
    val = os.environ.get(key, "")
    return val or default


ENGINE = get_env("ENGINE", "default").lower()
TEI_ENDPOINT = get_env("TEI_ENDPOINT", None)
TASKS = get_env("TASKS", "llm").lower().split(",")  # llm, rag

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
STORAGE_LOCAL_PATH = get_env(
    "STORAGE_LOCAL_PATH",
    os.path.join(Path(__file__).parents[1], "data", "file_storage")
)
os.makedirs(STORAGE_LOCAL_PATH, exist_ok=True)


<<<<<<< HEAD

class BaseSettings(BaseModel):
    """ Settings class. """

    host: Optional[str] = Field(
        default=get_env("HOST", "0.0.0.0"),  # 默认监听地址
        description="Listen address."  # 监听地址描述
    )

    port: Optional[int] = Field(
        default=int(get_env("PORT", 8000)),  # 默认监听端口
        description="Listen port."  # 监听端口描述
    )

    api_prefix: Optional[str] = Field(
        default=get_env("API_PREFIX", "/v1"),  # 默认API前缀
        description="API prefix."  # API前缀描述
    )

    engine: Optional[str] = Field(
        default=ENGINE,  # 默认引擎类型，可选值为['default', 'vllm', 'llama.cpp', 'tgi']
        description="Choices are ['default', 'vllm', 'llama.cpp', 'tgi']."  # 引擎选择描述
    )

    tasks: Optional[List[str]] = Field(
        default=list(TASKS),  # 默认任务列表，可选值为['llm', 'rag']
        description="Choices are ['llm', 'rag']."  # 任务选择描述
    )

    # device related
    device_map: Optional[Union[str, Dict]] = Field(
        default=get_env("DEVICE_MAP", "auto"),  # 默认设备映射配置
        description="Device map to load the model."  # 加载模型的设备映射描述
    )

    gpus: Optional[str] = Field(
        default=get_env("GPUS", None),  # 默认GPU配置
        description="Specify which gpus to load the model."  # 指定加载模型的GPU描述
    )

    num_gpus: Optional[int] = Field(
        default=int(get_env("NUM_GPUs", 1)),  # 默认GPU数量
        ge=0,  # 最小值为0
        description="How many gpus to load the model."  # 加载模型的GPU数量描述
    )

    activate_inference: Optional[bool] = Field(
        default=get_bool_env("ACTIVATE_INFERENCE", "true"),  # 默认是否激活推理
        description="Whether to activate inference."  # 是否激活推理描述
    )

    model_names: Optional[List] = Field(
        default_factory=list,  # 默认模型名称列表
        description="All available model names"  # 所有可用模型名称描述
    )

    # support for api key check
    api_keys: Optional[List[str]] = Field(
        default=get_env("API_KEYS", "").split(",") if get_env("API_KEYS", "") else None,  # 默认API密钥列表
        description="Support for api key check."  # 支持API密钥检查描述
=======
class BaseSettings(BaseModel):
    """ Settings class. """
    host: Optional[str] = Field(
        default=get_env("HOST", "0.0.0.0"),
        description="Listen address.",
    )
    port: Optional[int] = Field(
        default=int(get_env("PORT", 8000)),
        description="Listen port.",
    )
    api_prefix: Optional[str] = Field(
        default=get_env("API_PREFIX", "/v1"),
        description="API prefix.",
    )
    engine: Optional[str] = Field(
        default=ENGINE,
        description="Choices are ['default', 'vllm'].",
    )
    tasks: Optional[List[str]] = Field(
        default=list(TASKS),
        description="Choices are ['llm', 'rag'].",
    )
    # device related
    device_map: Optional[Union[str, Dict]] = Field(
        default=get_env("DEVICE_MAP", "auto"),
        description="Device map to load the model."
    )
    gpus: Optional[str] = Field(
        default=get_env("GPUS", None),
        description="Specify which gpus to load the model."
    )
    num_gpus: Optional[int] = Field(
        default=int(get_env("NUM_GPUs", 1)),
        ge=0,
        description="How many gpus to load the model."
    )
    activate_inference: Optional[bool] = Field(
        default=get_bool_env("ACTIVATE_INFERENCE", "true"),
        description="Whether to activate inference."
    )
    model_names: Optional[List] = Field(
        default_factory=list,
        description="All available model names"
    )
    # support for api key check
    api_keys: Optional[List[str]] = Field(
        default=get_env("API_KEYS", "").split(",") if get_env("API_KEYS", "") else None,
        description="Support for api key check."
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    )


class LLMSettings(BaseModel):
    # model related
    model_name: Optional[str] = Field(
<<<<<<< HEAD
        default=get_env("MODEL_NAME", None),  # 默认模型名称
        description="The name of the model to use for generating completions."  # 用于生成完成的模型名称描述
    )

    model_path: Optional[str] = Field(
        default=get_env("MODEL_PATH", None),  # 默认模型路径
        description="The path to the model to use for generating completions."  # 用于生成完成的模型路径描述
    )

    dtype: Optional[str] = Field(
        default=get_env("DTYPE", "half"),  # 默认数据类型精度
        description="Precision dtype."  # 数据类型精度描述
=======
        default=get_env("MODEL_NAME", None),
        description="The name of the model to use for generating completions."
    )
    model_path: Optional[str] = Field(
        default=get_env("MODEL_PATH", None),
        description="The path to the model to use for generating completions."
    )
    dtype: Optional[str] = Field(
        default=get_env("DTYPE", "half"),
        description="Precision dtype."
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    )

    # quantize related
    load_in_8bit: Optional[bool] = Field(
<<<<<<< HEAD
        default=get_bool_env("LOAD_IN_8BIT"),  # 默认是否以8位加载模型
        description="Whether to load the model in 8 bit."  # 是否以8位加载模型描述
    )

    load_in_4bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_4BIT"),  # 默认是否以4位加载模型
        description="Whether to load the model in 4 bit."  # 是否以4位加载模型描述
=======
        default=get_bool_env("LOAD_IN_8BIT"),
        description="Whether to load the model in 8 bit."
    )
    load_in_4bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_4BIT"),
        description="Whether to load the model in 4 bit."
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    )

    # context related
    context_length: Optional[int] = Field(
<<<<<<< HEAD
        default=int(get_env("CONTEXT_LEN", -1)),  # 默认上下文长度
        ge=-1,  # 大于等于-1
        description="Context length for generating completions."  # 生成完成的上下文长度描述
    )

    chat_template: Optional[str] = Field(
        default=get_env("PROMPT_NAME", None),  # 默认聊天模板名称
        description="Chat template for generating completions."  # 生成完成的聊天模板描述
    )

    rope_scaling: Optional[str] = Field(
        default=get_env("ROPE_SCALING", None),  # 默认RoPE缩放
        description="RoPE Scaling."  # RoPE缩放描述
    )

    flash_attn: Optional[bool] = Field(
        default=get_bool_env("FLASH_ATTN", "auto"),  # 默认是否使用闪光注意力
        description="Use flash attention."  # 使用闪光注意力描述
    )

    interrupt_requests: Optional[bool] = Field(
        default=get_bool_env("INTERRUPT_REQUESTS", "true"),  # 默认是否在接收到新请求时中断请求
        description="Whether to interrupt requests when a new request is received.",  # 在接收到新请求时是否中断请求描述
    )



class RAGSettings(BaseModel):
    # embedding related
    embedding_name: Optional[str] = Field(
        default=get_env("EMBEDDING_NAME", None),  # 默认嵌入模型名称
        description="The path to the model to use for generating embeddings."  # 用于生成嵌入的模型路径描述
    )

    rerank_name: Optional[str] = Field(
        default=get_env("RERANK_NAME", None),  # 默认重新排序模型名称
        description="The path to the model to use for reranking."  # 用于重新排序的模型路径描述
    )

    embedding_size: Optional[int] = Field(
        default=int(get_env("EMBEDDING_SIZE", -1)),  # 默认嵌入大小
        description="The embedding size to use for generating embeddings."  # 用于生成嵌入的嵌入大小描述
    )

    embedding_device: Optional[str] = Field(
        default=get_env("EMBEDDING_DEVICE", "cuda:0"),  # 默认加载嵌入模型的设备
        description="Device to load the model."  # 加载模型的设备描述
    )

    rerank_device: Optional[str] = Field(
        default=get_env("RERANK_DEVICE", "cuda:0"),  # 默认加载重新排序模型的设备
        description="Device to load the model."  # 加载模型的设备描述
    )



class VLLMSetting(BaseModel):
    # vllm related
    trust_remote_code: Optional[bool] = Field(
        default=get_bool_env("TRUST_REMOTE_CODE"),  # 是否信任远程代码
        description="Whether to use remote code."  # 是否使用远程代码描述
    )
    tokenize_mode: Optional[str] = Field(
        default=get_env("TOKENIZE_MODE", "auto"),  # 默认的标记化模式
        description="Tokenize mode for vllm server."  # VLLM服务器的标记化模式描述
    )
    tensor_parallel_size: Optional[int] = Field(
        default=int(get_env("TENSOR_PARALLEL_SIZE", 1)),  # 张量并行大小
        ge=1,
        description="Tensor parallel size for vllm server."  # VLLM服务器的张量并行大小描述
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=float(get_env("GPU_MEMORY_UTILIZATION", 0.9)),  # GPU内存利用率
        description="GPU memory utilization for vllm server."  # VLLM服务器的GPU内存利用率描述
    )
    max_num_batched_tokens: Optional[int] = Field(
        default=int(get_env("MAX_NUM_BATCHED_TOKENS", -1)),  # 最大批处理标记数
        ge=-1,
        description="Max num batched tokens for vllm server."  # VLLM服务器的最大批处理标记数描述
    )
    max_num_seqs: Optional[int] = Field(
        default=int(get_env("MAX_NUM_SEQS", 256)),  # 最大序列数
        ge=1,
        description="Max num seqs for vllm server."  # VLLM服务器的最大序列数描述
    )
    quantization_method: Optional[str] = Field(
        default=get_env("QUANTIZATION_METHOD", None),  # 量化方法
        description="Quantization method for vllm server."  # VLLM服务器的量化方法描述
    )
    enforce_eager: Optional[bool] = Field(
        default=get_bool_env("ENFORCE_EAGER"),  # 强制使用急切模式
        description="Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility."  # 总是使用急切模式的描述
    )
    max_seq_len_to_capture: Optional[int] = Field(
        default=int(get_env("MAX_SEQ_LEN_TO_CAPTURE", 8192)),  # 最大捕获的序列长度
        description="Maximum context length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode."  # 被CUDA图表覆盖的最大上下文长度描述
    )
    max_loras: Optional[int] = Field(
        default=int(get_env("MAX_LORAS", 1)),  # 最大LoRA数
        description="Max number of LoRAs in a single batch."  # 单个批次中的最大LoRA数描述
    )
    max_lora_rank: Optional[int] = Field(
        default=int(get_env("MAX_LORA_RANK", 32)),  # 最大LoRA秩
        description="Max LoRA rank."  # 最大LoRA秩描述
    )
    lora_extra_vocab_size: Optional[int] = Field(
        default=int(get_env("LORA_EXTRA_VOCAB_SIZE", 256)),  # 额外LoRA词汇的最大大小
        description="Maximum size of extra vocabulary that can be present in a LoRA adapter added to the base model vocabulary."  # 可添加到基础模型词汇表中的额外词汇的最大大小描述
    )
    lora_dtype: Optional[str] = Field(
        default=get_env("LORA_DTYPE", "auto"),  # LoRA数据类型
        description="Data type for LoRA. If auto, will default to base model dtype."  # LoRA的数据类型描述。如果是auto，将默认为基础模型数据类型
    )
    max_cpu_loras: Optional[int] = Field(
        default=int(get_env("MAX_CPU_LORAS", -1)),  # 最大CPU LoRA数
        ge=-1,
    )
    lora_modules: Optional[str] = Field(
        default=get_env("LORA_MODULES", ""),  # LoRA模块
    )
    disable_custom_all_reduce: Optional[bool] = Field(
        default=get_bool_env("DISABLE_CUSTOM_ALL_REDUCE"),  # 禁用自定义全局归约
    )
    vllm_disable_log_stats: Optional[bool] = Field(
        default=get_bool_env("VLLM_DISABLE_LOG_STATS", "true"),  # 禁用VLLM日志统计
    )


class LlamaCppSetting(BaseModel):
    n_gpu_layers: Optional[int] = Field(
        default=int(get_env("N_GPU_LAYERS", 0)),  # GPU层的数量
        ge=-1,
        description="The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU.",  # 放置在GPU上的层的数量描述。其余将放在CPU上。设置为-1以全部移至GPU
    )
    main_gpu: Optional[int] = Field(
        default=int(get_env("MAIN_GPU", 0)),  # 主GPU使用
        ge=0,
        description="Main GPU to use.",  # 要使用的主GPU描述
    )
    tensor_split: Optional[List[float]] = Field(
        default=float(get_env("TENSOR_SPLIT", None)) if get_env("TENSOR_SPLIT", None) else None,  # 张量分割
        description="Split layers across multiple GPUs in proportion.",  # 在多个GPU之间按比例分割层描述
    )
    n_batch: Optional[int] = Field(
        default=int(get_env("N_BATCH", 512)),  # 每次评估使用的批次大小
        ge=1,
        description="The batch size to use per eval."  # 每次评估使用的批次大小描述
    )
    n_threads: Optional[int] = Field(
        default=int(get_env("N_THREADS", max(multiprocessing.cpu_count() // 2, 1))),  # 使用的线程数
        ge=1,
        description="The number of threads to use.",  # 使用的线程数描述
    )
    n_threads_batch: Optional[int] = Field(
        default=int(get_env("N_THREADS_BATCH", max(multiprocessing.cpu_count() // 2, 1))),  # 批处理时使用的线程数
        ge=0,
        description="The number of threads to use when batch processing.",  # 批处理时使用的线程数描述
    )
    rope_scaling_type: Optional[int] = Field(
        default=int(get_env("ROPE_SCALING_TYPE", -1))
    )
    rope_freq_base: Optional[float] = Field(
        default=float(get_env("ROPE_FREQ_BASE", 0.0)),  # RoPE基础频率
        description="RoPE base frequency"  # RoPE基础频率描述
    )
    rope_freq_scale: Optional[float] = Field(
        default=float(get_env("ROPE_FREQ_SCALE", 0.0)),  # RoPE频率缩放因子
        description="RoPE frequency scaling factor",  # RoPE频率缩放因子描述
    )



class TGISetting(BaseSettings):
    # support for tgi: https://github.com/huggingface/text-generation-inference
    tgi_endpoint: Optional[str] = Field(
        default=get_env("TGI_ENDPOINT", None),  # TGI端点，默认为环境变量中的值，若未设置则为None
        description="Text Generation Inference Endpoint.",  # 文本生成推理端点描述
=======
        default=int(get_env("CONTEXT_LEN", -1)),
        ge=-1,
        description="Context length for generating completions."
    )
    chat_template: Optional[str] = Field(
        default=get_env("PROMPT_NAME", None),
        description="Chat template for generating completions."
    )

    rope_scaling: Optional[str] = Field(
        default=get_env("ROPE_SCALING", None),
        description="RoPE Scaling."
    )
    flash_attn: Optional[bool] = Field(
        default=get_bool_env("FLASH_ATTN", "auto"),
        description="Use flash attention."
    )

    interrupt_requests: Optional[bool] = Field(
        default=get_bool_env("INTERRUPT_REQUESTS", "true"),
        description="Whether to interrupt requests when a new request is received.",
    )


class RAGSettings(BaseModel):
    # embedding related
    embedding_name: Optional[str] = Field(
        default=get_env("EMBEDDING_NAME", None),
        description="The path to the model to use for generating embeddings."
    )
    rerank_name: Optional[str] = Field(
        default=get_env("RERANK_NAME", None),
        description="The path to the model to use for reranking."
    )
    embedding_size: Optional[int] = Field(
        default=int(get_env("EMBEDDING_SIZE", -1)),
        description="The embedding size to use for generating embeddings."
    )
    embedding_device: Optional[str] = Field(
        default=get_env("EMBEDDING_DEVICE", "cuda:0"),
        description="Device to load the model."
    )
    rerank_device: Optional[str] = Field(
        default=get_env("RERANK_DEVICE", "cuda:0"),
        description="Device to load the model."
    )


class VLLMSetting(BaseModel):
    trust_remote_code: Optional[bool] = Field(
        default=get_bool_env("TRUST_REMOTE_CODE"),
        description="Whether to use remote code."
    )
    tokenize_mode: Optional[str] = Field(
        default=get_env("TOKENIZE_MODE", "auto"),
        description="Tokenize mode for vllm server."
    )
    tensor_parallel_size: Optional[int] = Field(
        default=int(get_env("TENSOR_PARALLEL_SIZE", 1)),
        ge=1,
        description="Tensor parallel size for vllm server."
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=float(get_env("GPU_MEMORY_UTILIZATION", 0.9)),
        description="GPU memory utilization for vllm server."
    )
    max_num_batched_tokens: Optional[int] = Field(
        default=int(get_env("MAX_NUM_BATCHED_TOKENS", -1)),
        ge=-1,
        description="Max num batched tokens for vllm server."
    )
    max_num_seqs: Optional[int] = Field(
        default=int(get_env("MAX_NUM_SEQS", 256)),
        ge=1,
        description="Max num seqs for vllm server."
    )
    quantization_method: Optional[str] = Field(
        default=get_env("QUANTIZATION_METHOD", None),
        description="Quantization method for vllm server."
    )
    enforce_eager: Optional[bool] = Field(
        default=get_bool_env("ENFORCE_EAGER"),
        description="Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility."
    )
    max_seq_len_to_capture: Optional[int] = Field(
        default=int(get_env("MAX_SEQ_LEN_TO_CAPTURE", 8192)),
        description="Maximum context length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode."
    )
    max_loras: Optional[int] = Field(
        default=int(get_env("MAX_LORAS", 1)),
        description="Max number of LoRAs in a single batch."
    )
    max_lora_rank: Optional[int] = Field(
        default=int(get_env("MAX_LORA_RANK", 32)),
        description="Max LoRA rank."
    )
    lora_extra_vocab_size: Optional[int] = Field(
        default=int(get_env("LORA_EXTRA_VOCAB_SIZE", 256)),
        description="Maximum size of extra vocabulary that can be present in a LoRA adapter added to the base model vocabulary."
    )
    lora_dtype: Optional[str] = Field(
        default=get_env("LORA_DTYPE", "auto"),
        description="Data type for LoRA. If auto, will default to base model dtype."
    )
    max_cpu_loras: Optional[int] = Field(
        default=int(get_env("MAX_CPU_LORAS", -1)),
        ge=-1,
    )
    lora_modules: Optional[str] = Field(
        default=get_env("LORA_MODULES", ""),
    )
    disable_custom_all_reduce: Optional[bool] = Field(
        default=get_bool_env("DISABLE_CUSTOM_ALL_REDUCE"),
    )
    vllm_disable_log_stats: Optional[bool] = Field(
        default=get_bool_env("VLLM_DISABLE_LOG_STATS", "true"),
    )
    distributed_executor_backend: Optional[str] = Field(
        default=get_env("DISTRIBUTED_EXECUTOR_BACKEND", None),
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    )


TEXT_SPLITTER_CONFIG = {
    "ChineseRecursiveTextSplitter": {
<<<<<<< HEAD
        "source": "huggingface",   # 使用huggingface作为来源
        "tokenizer_name_or_path": get_env("EMBEDDING_NAME", ""),  # 标记器名称或路径，默认为环境变量中的值，若未设置则为空字符串
    },
    "SpacyTextSplitter": {
        "source": "huggingface",  # 使用huggingface作为来源
        "tokenizer_name_or_path": "gpt2",  # 标记器名称或路径为"gpt2"
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",  # 使用tiktoken作为来源
        "tokenizer_name_or_path": "cl100k_base",  # 标记器名称或路径为"cl100k_base"
=======
        "source": "huggingface",   # 选择tiktoken则使用openai的方法
        "tokenizer_name_or_path": get_env("EMBEDDING_NAME", ""),
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on":
            [
<<<<<<< HEAD
                ("#", "head1"),    # 头部分割标记为"#", 别名为"head1"
=======
                ("#", "head1"),
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                ("##", "head2"),
                ("###", "head3"),
                ("####", "head4"),
            ]
    },
}


<<<<<<< HEAD
PARENT_CLASSES = [BaseSettings]  # 父类列表，初始包含BaseSettings

if "llm" in TASKS:  # 如果"llm"任务在TASKS列表中
    if ENGINE == "default":  # 如果引擎为"default"
        PARENT_CLASSES.append(LLMSettings)  # 添加LLMSettings到父类列表
    elif ENGINE == "vllm":  # 如果引擎为"vllm"
        PARENT_CLASSES.extend([LLMSettings, VLLMSetting])  # 添加LLMSettings和VLLMSetting到父类列表
    elif ENGINE == "llama.cpp":  # 如果引擎为"llama.cpp"
        PARENT_CLASSES.extend([LLMSettings, LlamaCppSetting])  # 添加LLMSettings和LlamaCppSetting到父类列表
    elif ENGINE == "tgi":  # 如果引擎为"tgi"
        PARENT_CLASSES.extend([LLMSettings, TGISetting])  # 添加LLMSettings和TGISetting到父类列表

if "rag" in TASKS:  # 如果"rag"任务在TASKS列表中
    PARENT_CLASSES.append(RAGSettings)  # 添加RAGSettings到父类列表
=======
PARENT_CLASSES = [BaseSettings]

if "llm" in TASKS:
    if ENGINE == "default":
        PARENT_CLASSES.append(LLMSettings)
    elif ENGINE == "vllm":
        PARENT_CLASSES.extend([LLMSettings, VLLMSetting])

if "rag" in TASKS:
    PARENT_CLASSES.append(RAGSettings)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d


class Settings(*PARENT_CLASSES):
    ...


<<<<<<< HEAD
SETTINGS = Settings()  # 创建Settings实例
for name in ["model_name", "embedding_name", "rerank_name"]:
    if getattr(SETTINGS, name, None):  # 如果SETTINGS中的属性名存在
        SETTINGS.model_names.append(getattr(SETTINGS, name).split("/")[-1])  # 将属性值的最后一部分添加到model_names列表中
logger.debug(f"SETTINGS: {jsonify(SETTINGS, indent=4)}")  # 记录SETTINGS的JSON表示，缩进4个空格


if SETTINGS.gpus:  # 如果SETTINGS中存在gpus属性
    if len(SETTINGS.gpus.split(",")) < SETTINGS.num_gpus:  # 如果gpus属性分割后的长度小于num_gpus
        raise ValueError(
            f"Larger --num_gpus ({SETTINGS.num_gpus}) than --gpus {SETTINGS.gpus}!"
        )  # 抛出数值错误异常，提示--num_gpus大于--gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = SETTINGS.gpus  # 设置CUDA_VISIBLE_DEVICES环境变量为SETTINGS中的gpus属性值

=======
SETTINGS = Settings()
for name in ["model_name", "embedding_name", "rerank_name"]:
    if getattr(SETTINGS, name, None):
        SETTINGS.model_names.append(getattr(SETTINGS, name).split("/")[-1])
logger.debug(f"SETTINGS: {jsonify(SETTINGS, indent=4)}")


if SETTINGS.gpus:
    if len(SETTINGS.gpus.split(",")) < SETTINGS.num_gpus:
        raise ValueError(
            f"Larger --num_gpus ({SETTINGS.num_gpus}) than --gpus {SETTINGS.gpus}!"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = SETTINGS.gpus
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
