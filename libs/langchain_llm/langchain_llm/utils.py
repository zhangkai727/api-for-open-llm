from typing import Optional

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_lora(
    base_model_path: str,
    lora_path: str,
    target_model_path: str,
    max_shard_size: Optional[str] = "2GB",
    safe_serialization: Optional[bool] = True,
):
    # 应用 LoRA 方法将 LoRA 适配器合并到基础模型中，并保存目标模型和分词器。

    logger.info(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 使用 torch.float16 数据类型
        low_cpu_mem_usage=True,  # 低 CPU 内存使用
        trust_remote_code=True,  # 信任远程代码
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,  # 禁用快速模式
        trust_remote_code=True,  # 信任远程代码
    )

    logger.info(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(base, lora_path)  # 使用 PeftModel 加载 LoRA 适配器

    logger.info("Applying the LoRA")
    model = lora_model.merge_and_unload()  # 合并并卸载 LoRA 适配器

    logger.info(f"Saving the target model to {target_model_path}")
    model.save_pretrained(
        target_model_path,
        max_shard_size=max_shard_size,  # 最大碎片大小
        safe_serialization=safe_serialization,  # 安全序列化
    )
    base_tokenizer.save_pretrained(target_model_path)  # 保存分词器到目标模型路径

