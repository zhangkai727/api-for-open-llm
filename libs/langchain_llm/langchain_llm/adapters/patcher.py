""" from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/model/patcher.py s"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
)

import torch
from loguru import logger
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
)
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_npu_available
)
from transformers.utils.versions import require_version

if TYPE_CHECKING:  # 如果是类型检查
    from transformers import PretrainedConfig, PreTrainedTokenizer  # 导入类型提示的预训练配置和预训练分词器

_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()  # 检查是否支持fp16，NPU或CUDA可用
try:
    _is_bf16_available = is_torch_bf16_gpu_available()  # 尝试检查是否支持bf16
except:
    _is_bf16_available = False  # 如果出错，则设置为False，不支持bf16

def is_package_available(name: str) -> bool:
    """
    检查指定名称的包是否可用。
    """
    return importlib.util.find_spec(name) is not None  # 使用importlib检查包是否可用

def get_package_version(name: str) -> str:
    """
    获取指定包的版本号。
    """
    try:
        return importlib.metadata.version(name)  # 使用importlib获取包的版本号
    except:
        return "0.0.0"  # 获取失败时返回"0.0.0"

def is_flash_attn2_available():
    """
    检查是否安装了FlashAttention2。
    """
    return is_package_available("flash_attn") and get_package_version("flash_attn").startswith("2")  # 检查是否安装了FlashAttention2

def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    """
    根据模型dtype和设备兼容性推断最优dtype。
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16  # 如果支持bf16并且模型dtype是bfloat16，则返回bfloat16
    elif _is_fp16_available:
        return torch.float16  # 如果支持fp16，则返回float16
    else:
        return torch.float32  # 否则返回float32

def _configure_rope(config: "PretrainedConfig", rope_scaling: str = None) -> None:
    """
    配置RoPE（Relative Position Encoding）。
    """
    if not hasattr(config, "rope_scaling"):
        logger.warning("Current model does not support RoPE scaling.")  # 如果模型不支持RoPE缩放，则记录警告信息并返回
        return

    scaling_factor = 2.0  # 设置缩放因子为2.0
    setattr(config, "rope_scaling", {"type": rope_scaling, "factor": scaling_factor})  # 设置RoPE缩放策略和缩放因子
    logger.info(f"Using {rope_scaling} scaling strategy and setting scaling factor to {scaling_factor}.")  # 记录使用的RoPE缩放策略和缩放因子信息

def _configure_flashattn(config_kwargs: Dict[str, Any]) -> None:
    """
    配置FlashAttention-2。
    """
    if not is_flash_attn2_available():
        logger.warning("FlashAttention2 is not installed.")  # 如果未安装FlashAttention2，则记录警告信息并返回
        return

    config_kwargs["use_flash_attention_2"] = True  # 设置使用FlashAttention-2
    logger.info("Using FlashAttention-2 for faster and inference.")  # 记录使用FlashAttention-2进行更快的推理信息

def _configure_quantization(
    config_kwargs: Dict[str, Any],
    load_in_8bits: bool = False,
    load_in_4bits: bool = False,
) -> None:
    """
    配置模型量化。
    """
    if load_in_8bits:
        require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")  # 要求版本号大于等于0.37.0的bitsandbytes
        config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)  # 设置模型加载为8位
        logger.info("Quantizing model to 8 bit.")  # 记录模型量化为8位信息

    elif load_in_4bits:
        require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")  # 要求版本号大于等于0.39.0的bitsandbytes
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config_kwargs.get("torch_dtype", torch.float16),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )  # 设置模型加载为4位
        logger.info("Quantizing model to 4 bit.")  # 记录模型量化为4位信息

    if load_in_8bits or load_in_4bits:
        config_kwargs["device_map"] = {"": get_current_device()}  # 设置设备映射为当前设备
    else:
        config_kwargs["device_map"] = get_current_device()  # 设置设备映射为当前设备

def patch_tokenizer(tokenizer: "PreTrainedTokenizer") -> None:
    """
    对分词器进行补丁操作。
    """
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)  # 使用MethodType对分词器的_pad方法进行补丁

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = ""  # 如果分词器没有设置eos_token_id，则设置eos_token为空字符串
        logger.info(f"Add eos token: {tokenizer.eos_token}")  # 记录添加的eos token信息

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token  # 如果分词器的unk_token_id不为空，则设置pad_token为unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token  # 否则设置pad_token为eos_token
        logger.info(f"Add pad token: {tokenizer.pad_token}")  # 记录添加的pad token信息

def patch_config(
    config: "PretrainedConfig",
    config_kwargs: Dict[str, Any],
    compute_dtype: Optional[str] = None,
    **kwargs,
):
    """
    对配置进行补丁操作。
    """
    if compute_dtype is None:  # 如果compute_dtype为空，默认优先级：bf16 > fp16 > fp32
        compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))  # 推断最优dtype
    else:
        _DTYPE_MAP = {
            "half": torch.float16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = _DTYPE_MAP.get(compute_dtype, torch.float16)  # 根据指定的compute_dtype获取对应的dtype

    config_kwargs["torch_dtype"] = compute_dtype  # 设置torch_dtype为推断出的compute_dtype

    if getattr(config, "model_type", None) == "qwen":  # 如果配置的模型类型为"qwen"
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, compute_dtype == dtype)  # 设置fp16、bf16、fp32属性，根据compute_dtype的值

    rope_scaling = kwargs.get("rope_scaling", None)  # 获取rope_scaling参数
    if rope_scaling is not None:
        _configure_rope(config, rope_scaling)  # 配置RoPE

    if kwargs.get("flash_attn", False):  # 如果启用了flash_attn
        _configure_flashattn(config_kwargs)  # 配置FlashAttention-2

    _configure_quantization(  # 配置模型量化
        config_kwargs,
        kwargs.get("load_in_8bit", False),
        kwargs.get("load_in_4bit", False),
    )

def patch_model(model: "PreTrainedModel") -> None:
    """
    对模型进行补丁操作。
    """
    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)  # 使用MethodType对模型的generate方法进行补丁

def get_current_device() -> torch.device:
    """
    获取当前可用设备。
    """
    if is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))  # 如果NPU可用，则设置设备为npu
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))  # 如果CUDA

