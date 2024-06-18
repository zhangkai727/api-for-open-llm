""" from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/model/patcher.py """
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

<<<<<<< HEAD
if TYPE_CHECKING:       #TYPE_CHECKING 是一个标准库 typing 中的常量，它在类型检查上下文中为真。在运行时，TYPE_CHECKING 为假；在类型检查时（例如使用 mypy 这样的静态类型检查工具），TYPE_CHECKING 会被设置为真。
    from transformers import PretrainedConfig, PreTrainedTokenizer      #TYPE_CHECKING 是一个标准库 typing 中的常量，它在类型检查上下文中为真。在运行时，TYPE_CHECKING 为假；在类型检查时（例如使用 mypy 这样的静态类型检查工具），TYPE_CHECKING 会被设置为真。


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()  #这行代码定义了一个变量 _is_fp16_available，用来表示当前环境是否可用于FP16精度的计算。
try:        #一个异常处理块，用于捕获可能发生的异常
    _is_bf16_available = is_torch_bf16_gpu_available()         #在 try 块中，调用了 is_torch_bf16_gpu_available() 函数，用来检查当前环境是否支持BF16（BFloat16）精度的GPU计算。
except:
    _is_bf16_available = False      #将 _is_bf16_available 设置为 False，表示BF16计算不可用。


def is_package_available(name: str) -> bool:        #定义了一个函数 is_package_available，它接受一个字符串类型的参数 name,函数将返回一个布尔值，表明指定的包是否可用
    return importlib.util.find_spec(name) is not None       #如果找到了指定名称的包或模块，则 find_spec(name) 返回一个 ModuleSpec 对象；否则返回 None。


def get_package_version(name: str) -> str:      #定义了一个函数 get_package_version，它接受一个字符串类型的参数 name,函数将返回一个字符串
    try:
        return importlib.metadata.version(name)     #如果成功获取到指定包或模块的版本号，则函数返回该版本号的字符串形式
    except:
        return "0.0.0"      #在发生异常时，函数返回字符串 "0.0.0"，表示未能获取到有效的版本号


def is_flash_attn2_available():     #定义了一个名为 is_flash_attn2_available 的函数，没有参数传入
    return is_package_available("flash_attn") and get_package_version("flash_attn").startswith("2")
            #调用之前定义的 is_package_available 函数，检查名为 "flash_attn" 的包是否可用和调用之前定义的 get_package_version 函数，获取名为 "flash_attn" 的包的版本号是否以 "2" 开头。

def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:     #infer_optim_dtype 是函数名，接受一个参数 model_dtype，类型为 torch.dtype，并且返回值也是 torch.dtype 类型。
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.  函数的目的是根据 model_dtype 和设备的兼容性推断出最优的数据类型。
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16   #如果变量 _is_bf16_available 可用（表示当前环境支持 torch.bfloat16）并且输入的 model_dtype 等于 torch.bfloat16，则返回 torch.bfloat16
    elif _is_fp16_available:
        return torch.float16    #如果前一个条件不满足，检查变量 _is_fp16_available 是否可用（表示当前环境支持 torch.float16），如果是，则返回 torch.float16。
    else:
        return torch.float32    #如果前两个条件都不满足，则返回默认的数据类型 torch.float32。


def _configure_rope(config: "PretrainedConfig", rope_scaling: str = None) -> None:#_configure_rope 是函数名，接受两个参数，config：类型为 "PretrainedConfig"，表示预训练配置对象，    ope_scaling：可选参数，类型为 str，用于指定 RoPE 缩放策略，默认为 None。函数没有返回值，因为返回类型注解为 None。
    if not hasattr(config, "rope_scaling"): #首先检查 config 对象是否具有 rope_scaling 属性。如果没有这个属性，说明当前模型不支持 RoPE 缩放。
        logger.warning("Current model does not support RoPE scaling.")
        return

    scaling_factor = 2.0    #如果模型支持 RoPE 缩放，那么将设置 config 对象的 rope_scaling 属性。这里使用了 setattr 函数，将属性名设为 "rope_scaling"，并赋予一个字典作为属性值，字典包含两个键值对
    setattr(config, "rope_scaling", {"type": rope_scaling, "factor": scaling_factor})
    logger.info(f"Using {rope_scaling} scaling strategy and setting scaling factor to {scaling_factor}.")
    #记录信息日志，说明使用了哪种 RoPE 缩放策略（即 rope_scaling 的值）和设置的缩放因子值。

def _configure_flashattn(config_kwargs: Dict[str, Any]) -> None:    #接受一个参数 config_kwargs，类型为 Dict[str, Any]，表示配置参数的字典。函数没有返回值，因为返回类型注解为 None
    if not is_flash_attn2_available():  #用于检查当前环境是否安装了 FlashAttention-2。如果返回 False，表示 FlashAttention-2 没有安装。
        logger.warning("FlashAttention2 is not installed.")
        return

    config_kwargs["use_flash_attention_2"] = True   #如果 FlashAttention_2 安装可用，将在 config_kwargs 字典中添加一个键值对
    logger.info("Using FlashAttention-2 for faster and inference.")#记录信息日志，表明正在使用 FlashAttention-2 来加速推理过程。


def _configure_quantization(
    config_kwargs: Dict[str, Any],  #表示配置参数的字典
    load_in_8bits: bool = False,    #表示是否加载 8 位量化模型
    load_in_4bits: bool = False,    #表示是否加载 4 位量化模型
) -> None:

    if load_in_8bits:               #如果 load_in_8bits 参数为 True
        require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")#检查是否安装了 bitsandbytes 包的版本 0.37.0 或以上，如果未安装则输出错误信息。
        config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)#添加一个键为 "quantization_config" 的配置，值为 BitsAndBytesConfig(load_in_8bit=True)，表示将模型量化到 8 位
        logger.info("Quantizing model to 8 bit.")       #使用日志记录器 logger 输出信息，提示正在将模型量化到 8 位

    elif load_in_4bits:             #如果 load_in_4bits 参数为 True
        require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")#同上
        config_kwargs["quantization_config"] = BitsAndBytesConfig(#添加一个键为 "quantization_config" 的配置，值为 BitsAndBytesConfig 对象，
            load_in_4bit=True,                                      #表示将模型量化到 4 位
=======
if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except:
    _is_bf16_available = False


def is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except:
        return "0.0.0"


def is_flash_attn2_available():
    return is_package_available("flash_attn") and get_package_version("flash_attn").startswith("2")


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def _configure_rope(config: "PretrainedConfig", rope_scaling: str = None) -> None:
    if not hasattr(config, "rope_scaling"):
        logger.warning("Current model does not support RoPE scaling.")
        return

    scaling_factor = 2.0
    setattr(config, "rope_scaling", {"type": rope_scaling, "factor": scaling_factor})
    logger.info(f"Using {rope_scaling} scaling strategy and setting scaling factor to {scaling_factor}.")


def _configure_flashattn(config_kwargs: Dict[str, Any]) -> None:
    if not is_flash_attn2_available():
        logger.warning("FlashAttention2 is not installed.")
        return

    config_kwargs["use_flash_attention_2"] = True
    logger.info("Using FlashAttention-2 for faster and inference.")


def _configure_quantization(
    config_kwargs: Dict[str, Any],
    load_in_8bits: bool = False,
    load_in_4bits: bool = False,
) -> None:

    if load_in_8bits:
        require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
        config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Quantizing model to 8 bit.")

    elif load_in_4bits:
        require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            bnb_4bit_compute_dtype=config_kwargs.get("torch_dtype", torch.float16),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
<<<<<<< HEAD
        logger.info("Quantizing model to 4 bit.")#使用日志记录器 logger 输出信息，提示正在将模型量化到 4 位

    if load_in_8bits or load_in_4bits:      #果 load_in_8bits 或 load_in_4bits 中任何一个为 True，则执行下面的代码块
        config_kwargs["device_map"] = {"": get_current_device()}#如果上述条件成立，将 "device_map" 键的值设为 {"": get_current_device()}
    else:
        config_kwargs["device_map"] = get_current_device()#否则，就会将 "device_map" 设置为 get_current_device() 的返回值。


#检查 tokenizer 对象的 _pad 方法是否已经绑定到 PreTrainedTokenizerBase 类的 _pad 方法。如果没有，则将 PreTrainedTokenizerBase._pad 方法绑定到 tokenizer._pad 上，以确保 tokenizer 具备正确的填充方法。
=======
        logger.info("Quantizing model to 4 bit.")

    if load_in_8bits or load_in_4bits:
        config_kwargs["device_map"] = {"": get_current_device()}
    else:
        config_kwargs["device_map"] = get_current_device()


>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
def patch_tokenizer(tokenizer: "PreTrainedTokenizer") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

<<<<<<< HEAD
    if tokenizer.eos_token_id is None:  #如果 tokenizer 的 eos_token_id 是 None，则将 eos_token 设置为空字符串 ""。并记录日志，表示已添加了 eos_token
        tokenizer.eos_token = "<|endoftext|>"
        logger.info(f"Add eos token: {tokenizer.eos_token}")


    #如果 tokenizer 的 pad_token_id 是 None，则根据情况设置 pad_token，如果 unk_token_id 不为 None，则将 pad_token 设置为 unk_token
    #否则，将 pad_token 设置为 eos_token，记录日志，表示已添加了 pad_token
=======
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info(f"Add eos token: {tokenizer.eos_token}")

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad token: {tokenizer.pad_token}")


<<<<<<< HEAD
def patch_config(#定义一个函数
    config: "PretrainedConfig",#类型为 PretrainedConfig 的对象，表示预训练配置。
    config_kwargs: Dict[str, Any],#为 Dict[str, Any] 的字典，包含用于配置的关键字参数
    compute_dtype: Optional[str] = None,  #使用Optional参数可以是 None 或指定的类型。
    **kwargs,       #用于接收额外的关键字参数，
):
    if compute_dtype is None:  # priority: bf16 > fp16 > fp32
        compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))#调用 infer_optim_dtype 函数，从配置对象 config 中获取 torch_dtype 属性的值（如果有的话），作为 model_dtype 参数传入。
    else:
        _DTYPE_MAP = {  #如果 compute_dtype 不是 None，则根据 _DTYPE_MAP 字典将 compute_dtype 转换为对应的 PyTorch 数据类型：
=======
def patch_config(
    config: "PretrainedConfig",
    config_kwargs: Dict[str, Any],
    compute_dtype: Optional[str] = None,
    **kwargs,
):
    if compute_dtype is None:  # priority: bf16 > fp16 > fp32
        compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))
    else:
        _DTYPE_MAP = {
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            "half": torch.float16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
<<<<<<< HEAD
        compute_dtype = _DTYPE_MAP.get(compute_dtype, torch.float16)#如果 compute_dtype 不在 _DTYPE_MAP 中，则默认使用 torch.float16

    config_kwargs["torch_dtype"] = compute_dtype#将compute_dtype 存储在 config_kwargs 字典中的 torch_dtype 键中，

    if getattr(config, "model_type", None) == "qwen":#配置对象 config 的 model_type 属性是否为 "qwen" 来设置不同的数据类型标志位
        #循环遍历列表 [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)] 中的每对元组 (dtype_name, dtype)
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, compute_dtype == dtype)

    rope_scaling = kwargs.get("rope_scaling", None)#从 kwargs 中获取 rope_scaling 参数的值，如果不存在则设为 None。
    if rope_scaling is not None:#如果 rope_scaling 参数不为 None，则调用 _configure_rope(config, rope_scaling) 函数，配置模型的 RoPE Scaling。
        _configure_rope(config, rope_scaling)

#如果 kwargs 中有设置 flash_attn 为 True，则调用 _configure_flashattn(config_kwargs) 函数，配置 FlashAttention。
    if kwargs.get("flash_attn", False):
        _configure_flashattn(config_kwargs)

#根据 kwargs 中的 load_in_8bit 和 load_in_4bit 参数配置量化选项。
=======
        compute_dtype = _DTYPE_MAP.get(compute_dtype, torch.float16)

    config_kwargs["torch_dtype"] = compute_dtype

    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, compute_dtype == dtype)

    rope_scaling = kwargs.get("rope_scaling", None)
    if rope_scaling is not None:
        _configure_rope(config, rope_scaling)

    if kwargs.get("flash_attn", False):
        _configure_flashattn(config_kwargs)

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
    _configure_quantization(
        config_kwargs,
        kwargs.get("load_in_8bit", False),
        kwargs.get("load_in_4bit", False),
    )


<<<<<<< HEAD
def patch_model(model: "PreTrainedModel") -> None:#接受一个参数 model，类型标注为 "PreTrainedModel"，返回类型被标注为 None，函数没有返回值。
    if "GenerationMixin" not in str(model.generate.__func__):#检查当前模型 model 的 generate 方法是否已经包含了名为 GenerationMixin 的方法
        model.generate = MethodType(PreTrainedModel.generate, model)#如果没有，需要进行补丁。


def get_current_device() -> torch.device:#定义一个函数，返回一个 torch.device 对象。
    r"""
    Gets the current available device.
    """
    if is_torch_npu_available():#是否有可用的NPU设备。
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))#如果有，则将 device 设为 "npu:{LOCAL_RANK}"，其中 LOCAL_RANK 是环境变量中的本地排名（local rank），如果未设置则默认为 "0"。
    elif is_torch_cuda_available():#如果没有NPU设备可用，继续检查是否有可用的CUDA设备
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))#如果有，则将 device 设为 "cuda:{LOCAL_RANK}"，同样使用环境变量中的本地排名，未设置则默认为 "0"如果有，则将 device 设为 "cuda:{LOCAL_RANK}"，同样使用环境变量中的本地排名，未设置则默认为 "0"
    else:
        device = "cpu"#如果系统既没有NPU设备也没有CUDA设备，则将 device 设为 "cpu"

    return torch.device(device)#最终确定的设备名称存储在变量 device 中，并作为函数的返回值
=======
def patch_model(model: "PreTrainedModel") -> None:
    if model.config.model_type == "minicpmv":
        return
    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)


def get_current_device() -> torch.device:
    r"""
    Gets the current available device.
    """
    if is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
