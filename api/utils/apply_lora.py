"""
Apply the LoRA weights on top of a base model.

Usage:
python api/utils/apply_lora.py --base ~/model_weights/llama-7b --target ~/model_weights/baize-7b --lora project-baize/baize-lora-7B
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_lora(base_model_path, target_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")  # 打印正在从base_model_path加载基础模型
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 使用torch.float16数据类型
        trust_remote_code=True,  # 信任远程代码
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)  # 从base_model_path加载基础分词器

    print(f"Loading the LoRA adapter from {lora_path}")  # 打印正在从lora_path加载LoRA适配器

    lora_model = PeftModel.from_pretrained(base, lora_path)  # 从base和lora_path加载LoRA模型

    print("Applying the LoRA")  # 打印正在应用LoRA

    model = lora_model.merge_and_unload()  # 合并并卸载LoRA模型

    print(f"Saving the target model to {target_model_path}")  # 打印正在保存目标模型到target_model_path
    model.save_pretrained(target_model_path, max_shard_size="10GB")  # 将模型保存到target_model_path，最大碎片大小为10GB
    base_tokenizer.save_pretrained(target_model_path)  # 将分词器保存到target_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--base-model-path", type=str, required=True)  # 添加必需的base-model-path参数
    parser.add_argument("--target-model-path", type=str, required=True)  # 添加必需的target-model-path参数
    parser.add_argument("--lora-path", type=str, required=True)  # 添加必需的lora-path参数

    args = parser.parse_args()  # 解析命令行参数

    apply_lora(args.base_model_path, args.target_model_path, args.lora_path)  # 应用LoRA到指定的模型路径

