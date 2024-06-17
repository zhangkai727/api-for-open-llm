from __future__ import annotations

from typing import (                    #从 Python 的 typing 模块中导入了几个类型和工具
    TYPE_CHECKING,                      #一个常量，通常用于在类型检查期间进行条件导入。它在运行时始终为 False，但在静态类型检查工具（如 MyPy）中为 True
    Optional,                           #一个泛型类型，用于表示可以是某种类型或者是 None 的值
    Tuple,                              #个泛型类型，用于表示固定长度和特定类型的元组
    Any,                                #一个特殊类型，表示任何类型。它可以用在不确定类型或者希望绕过类型检查的地方
)

from transformers import (              #从从 transformers 库中导入某些模块、类或函数
    AutoConfig,                         #AutoConfig 类用于自动加载模型的配置
    AutoModelForCausalLM,               #AutoModelForCausalLM 类用于加载因果语言模型（Causal Language Model），
    AutoTokenizer,                      #AutoTokenizer 类用于加载与特定模型相对应的标记器（Tokenizer）。标记器将输入文本分割成模型可以处理的子词或标记（tokens）。
)

from .patcher import (
    patch_config,                       #patch_config 是一个函数或对象，用于修补或修改配置对象。它可能用于调整模型的配置，例如修改模型架构或超参数。
    patch_tokenizer,                    #patch_tokenizer 是一个函数或对象，用于修补或修改标记器（Tokenizer）。它可能用于调整标记器的行为，例如添加新的词汇或修改标记化规则。
    patch_model,                        #patch_model 是一个函数或对象，用于修补或修改模型对象。它可能用于调整模型的参数或结构，例如添加新的层或修改现有层的配置。
)

if TYPE_CHECKING:                       #TYPE_CHECKING 是一个布尔常量，它在类型检查时（例如使用 mypy 等静态类型检查工具）被设为 True，而在运行时则为 False
    from transformers import PreTrainedModel, PreTrainedTokenizer#从 transformers 模块中导入 PreTrainedModel 和 PreTrainedTokenizer 类型。


def load_model_and_tokenizer(           #这段代码定义了一个名为 load_model_and_tokenizer 的函数，用于加载预训练模型和标记器（tokenizer）。它接受一系列参数来配置加载过程，并返回一个包含预训练模型和标记器的元组。
    model_name_or_path: str,            #这是一个字符串参数，用于指定模型的名称或路径。这个参数可以是 Hugging Face 模型库中的模型名称，也可以是本地保存的模型路径。
    use_fast_tokenizer: Optional[bool] = False,     #一个可选的布尔参数，指示是否使用快速标记器（fast tokenizer）。快速标记器通常比普通标记器更快，但有时在某些情况下可能不支持所有功能。
    dtype: Optional[str] = None,                #一个可选的字符串参数，指定模型加载时的数据类型。
    device_map: Optional[Any] = None,       #一个可选的参数，用于指定设备映射（device map）
    load_in_8bit: Optional[bool] = False,       #一个可选的布尔参数，指示是否将模型加载为 8 位精度，以减少内存占用
    load_in_4bit: Optional[bool] = False,       #一个可选的布尔参数，指示是否将模型加载为 4 位精度，以进一步减少内存占用
    rope_scaling: Optional[str] = None,     #一个可选的字符串参数，用于指定 ROPE（Rotary Position Embedding）缩放的方式
    flash_attn: Optional[bool] = False,     #一个可选的布尔参数，指示是否使用闪存注意力（flash attention）机制，这是一种优化的注意力机制，可以加速推理过程。
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:       #该函数返回一个包含预训练模型和标记器的元组。返回类型是通过字符串形式注解的，以避免在类型检查时导入实际类型。
    r"""
    Loads pretrained model and tokenizer.

    Support inference.
    """
    config_kwargs = {"trust_remote_code": True}     #这是一个配置参数字典，用于指定在加载模型配置时是否信任远程代码。设置 trust_remote_code 为 True 表示信任从 Hugging Face 模型库中下载的模型配置代码。这个选项在加载自定义配置或代码时非常有用。

    tokenizer = AutoTokenizer.from_pretrained(      #AutoTokenizer 是 Hugging Face transformers 库中的一个类，它提供了加载各种预训练标记器的功能。
        model_name_or_path,     #这是一个字符串参数，指定了模型的名称或路径。它可以是 Hugging Face 模型库中的模型名称，也可以是本地保存的模型路径。
        use_fast=use_fast_tokenizer,        #这是一个布尔参数，指示是否使用快速标记器。
        trust_remote_code=True,     #这是一个布尔参数，指示是否信任远程代码。如果设置为 True，则在加载模型和标记器时，会信任 Hugging Face 模型库中的远程代码。
    )
    patch_tokenizer(tokenizer)      #是一个函数，用于对加载的标记器对象进行修补或修改。具体的修补内容取决于 patch_tokenizer 函数的实现。

    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)      #AutoConfig 是 Hugging Face transformers 库中的一个类，用于加载模型的配置,调用形参model_name_or_path，**config_kwargs
    patch_config(      #是一个函数，用于对加载的配置对象进行修补或修改
        config,      #这是一个 AutoConfig 对象，表示从预训练模型加载的配置。
        config_kwargs,      #这是之前传递给 AutoConfig.from_pretrained 方法的配置参数字典
        dtype,      #这是一个可选参数，表示数据类型（例如 float32、float16）。它可以用来设置模型参数的数据类型，以便进行混合精度训练或推理。
        rope_scaling=rope_scaling,      #这是一个可选参数，表示相对位置编码（Relative Position Encodings, ROPE）的缩放因子。用于调整位置编码的比例。
        flash_attn=flash_attn,      #这是一个可选参数，表示是否启用 Flash Attention。Flash Attention 是一种加速注意力机制计算的方法。
        load_in_4bit=load_in_4bit,      #这是一个可选参数，表示是否以 4 位量化的形式加载模型。4 位量化可以大幅减少模型的内存占用。
        load_in_8bit=load_in_8bit,      #这是一个可选参数，表示是否以 8 位量化的形式加载模型。8 位量化是较为常见的量化方法，可以在减少内存占用的同时保持较好的模型精度。
    )

    if device_map:      #这行代码检查是否提供了 device_map 参数。
        config_kwargs["device_map"] = device_map      #如果 device_map 参数存在，则将其添加到 config_kwargs 字典中。

    model = AutoModelForCausalLM.from_pretrained(      #使用 AutoModelForCausalLM 类的 from_pretrained 方法来加载预训练模型。
        model_name_or_path,      #
        config=config,      #
        low_cpu_mem_usage=True,      #是一个布尔参数，表示是否在模型加载时尽量减少 CPU 内存的使用。
        **config_kwargs      #这是一个字典，包含了额外的配置参数。通过 **config_kwargs，这些参数会被传递给 from_pretrained 方法，用于配置加载过程中的一些选项。
    )

    patch_model(model)      #这行代码调用了 patch_model 函数，传入了 model 参数。
    model.eval()      #在 PyTorch 中，调用 eval() 方法会将模型设置为评估模式，

    return model, tokenizer      #这行代码返回修补后的模型对象 model 和相应的分词器对象 tokenizer
