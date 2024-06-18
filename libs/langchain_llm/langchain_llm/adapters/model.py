from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Optional,
    Tuple,
    Any,
)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .patcher import (
    patch_config,
    patch_tokenizer,
    patch_model,
)

<<<<<<< HEAD
if TYPE_CHECKING:  # 如果是类型检查
    from transformers import PreTrainedModel, PreTrainedTokenizer  # 导入类型提示的模型和分词器


def load_model_and_tokenizer(  # 定义加载模型和分词器的函数
        model_name_or_path: str,  # 模型名称或路径，字符串类型
        use_fast_tokenizer: Optional[bool] = False,  # 是否使用快速分词器，默认为False
        dtype: Optional[str] = None,  # 数据类型，默认为None
        device_map: Optional[Any] = None,  # 设备映射，默认为None
        load_in_8bit: Optional[bool] = False,  # 是否以8位加载，默认为False
        load_in_4bit: Optional[bool] = False,  # 是否以4位加载，默认为False
        rope_scaling: Optional[str] = None,  # 绳索缩放类型，默认为None
        flash_attn: Optional[bool] = False,  # 是否启用快闪注意，默认为False
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:  # 返回类型为元组，包含PreTrainedModel和PreTrainedTokenizer

    config_kwargs = {"trust_remote_code": True}  # 配置参数字典，信任远程代码为True

    tokenizer = AutoTokenizer.from_pretrained(  # 使用AutoTokenizer从预训练模型加载分词器
        model_name_or_path,  # 模型名称或路径
        use_fast=use_fast_tokenizer,  # 是否使用快速分词器
        trust_remote_code=True,  # 信任远程代码为True
    )
    patch_tokenizer(tokenizer)  # 对分词器进行补丁操作

    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)  # 使用AutoConfig从预训练模型加载配置
    patch_config(  # 对配置进行补丁操作
=======
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    use_fast_tokenizer: Optional[bool] = False,
    dtype: Optional[str] = None,
    device_map: Optional[Any] = None,
    load_in_8bit: Optional[bool] = False,
    load_in_4bit: Optional[bool] = False,
    rope_scaling: Optional[str] = None,
    flash_attn: Optional[bool] = False,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support inference.
    """
    config_kwargs = {"trust_remote_code": True}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=True,
    )
    patch_tokenizer(tokenizer)

    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    patch_config(
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        config,
        config_kwargs,
        dtype,
        rope_scaling=rope_scaling,
        flash_attn=flash_attn,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

<<<<<<< HEAD
    if device_map:  # 如果设备映射不为空
        config_kwargs["device_map"] = device_map  # 设置配置参数中的设备映射

    model = AutoModelForCausalLM.from_pretrained(  # 使用AutoModelForCausalLM从预训练模型加载模型
        model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,  # 低CPU内存使用
        **config_kwargs
    )

    patch_model(model)  # 对模型进行补丁操作
    model.eval()  # 设置模型为评估模式

    return model, tokenizer  # 返回加载的模型和分词器

=======
    if device_map:
        config_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        **config_kwargs
    )

    patch_model(model)
    model.eval()

    return model, tokenizer
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
