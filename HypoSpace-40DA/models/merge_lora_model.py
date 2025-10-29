#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA权重与基座模型合并脚本
用于将LoRA微调结果与Qwen3-8B基座模型合并为完整模型

运行代码：
conda activate hypospace
python merge_lora_model.py \
    --base_model_path "/opt/data/private/HypoSpace-40DA/models/qwen/Qwen3-8B" \
    --lora_adapter_path "/opt/data/private/HypoSpace-40DA/models/lora/v0-20251028-162136/checkpoint-72" \
    --output_path "/opt/data/private/HypoSpace-40DA/models/qwen/Qwen3-8B-merged" \
    --device "auto"
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_with_base_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    device: str = "auto"
):
    """
    将LoRA适配器权重与基座模型合并
    
    Args:
        base_model_path (str): 基座模型路径
        lora_adapter_path (str): LoRA适配器路径
        output_path (str): 合并后模型保存路径
        device (str): 加载模型的设备 ("auto", "cpu", "cuda")
    """
    print(f"开始合并模型...")
    print(f"基座模型路径: {base_model_path}")
    print(f"LoRA适配器路径: {lora_adapter_path}")
    print(f"输出路径: {output_path}")
    
    # 确定设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载基座模型
    print("正在加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    print("基座模型加载完成")
    
    # 加载tokenizer
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    print("Tokenizer加载完成")
    
    # 加载LoRA适配器
    print("正在加载LoRA适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        device_map=device
    )
    print("LoRA适配器加载完成")
    
    # 合并权重
    print("正在合并权重...")
    model = model.merge_and_unload()
    print("权重合并完成")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 保存合并后的模型
    print("正在保存合并后的模型...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"合并后的模型已保存到: {output_path}")
    
    # 验证保存的模型
    print("正在验证保存的模型...")
    try:
        test_model = AutoModelForCausalLM.from_pretrained(
            output_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="cpu"  # 使用CPU验证以避免显存问题
        )
        print("模型验证成功!")
        print(f"模型参数量: {test_model.num_parameters():,}")
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="合并LoRA适配器与基座模型")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="基座模型路径"
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="LoRA适配器路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="合并后模型保存路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="加载模型的设备"
    )
    
    args = parser.parse_args()
    
    # 检查输入路径是否存在
    if not os.path.exists(args.base_model_path):
        print(f"错误: 基座模型路径不存在: {args.base_model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.lora_adapter_path):
        print(f"错误: LoRA适配器路径不存在: {args.lora_adapter_path}")
        sys.exit(1)
    
    # 执行合并
    success = merge_lora_with_base_model(
        args.base_model_path,
        args.lora_adapter_path,
        args.output_path,
        args.device
    )
    
    if success:
        print("模型合并完成!")
    else:
        print("模型合并失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()