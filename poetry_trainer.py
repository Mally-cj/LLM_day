#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文诗歌生成模型训练和推理脚本
基于 GPT-2 模型，使用 Chinese-Poems 数据集进行微调
"""

import os
import torch
import json
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import argparse

class PoetryTrainer:
    def __init__(self, model_name="openai-community/gpt2", output_dir="./poetry_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和分词器
        print(f"正在加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"使用设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_and_explore_dataset(self, max_samples=None):
        """加载并探索数据集"""
        print("正在加载数据集...")
        self.ds = load_dataset("Million/Chinese-Poems")
        
        print(f"数据集结构: {self.ds}")
        print(f"原始训练集样本数: {len(self.ds['train'])}")
        
        # 如果指定了最大样本数，则只使用前N个样本
        if max_samples is not None:
            self.ds['train'] = self.ds['train'].select(range(min(max_samples, len(self.ds['train']))))
            print(f"使用训练样本数: {len(self.ds['train'])}")
        
        # 查看数据集内容
        print("\n数据集样本示例:")
        for i in range(3):
            sample = self.ds['train'][i]
            print(f"样本 {i+1}:")
            print(f"  instruction: {sample['instruction']}")
            print(f"  input: {sample['input']}")
            print(f"  output: {sample['output'][:100]}...")  # 只显示前100个字符
            print()
    
    def preprocess_function(self, examples):
        """数据预处理函数"""
        texts = []
        for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
            # 构建训练文本格式
            text = f"指令: {inst}\n要求: {inp}\n诗词: {out}<|endoftext|>"
            texts.append(text)
        
        # 分词
        tokenized = self.tokenizer(
            texts,
            max_length=512,  # 增加长度以容纳更长的诗歌
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建标签，只对诗词部分计算损失
        labels = tokenized["input_ids"].clone()
        for i, text in enumerate(texts):
            poetry_start = text.find("诗词:") + 3
            prefix_text = text[:poetry_start]
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            # 将前缀部分设为 -100，不计算损失
            labels[i, :len(prefix_tokens)+1] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
    
    def train(self, num_epochs=3, batch_size=4, learning_rate=5e-5, max_samples=None):
        """训练模型"""
        # 如果指定了最大样本数，先限制数据集大小
        if max_samples is not None:
            self.ds['train'] = self.ds['train'].select(range(min(max_samples, len(self.ds['train']))))
            print(f"使用训练样本数: {len(self.ds['train'])}")
        
        print("正在预处理数据...")
        tokenized_ds = self.ds.map(
            self.preprocess_function, 
            batched=True, 
            remove_columns=self.ds["train"].column_names
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),  # 使用混合精度训练
            dataloader_num_workers=4,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        print("开始训练...")
        print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        trainer.train()
        
        # 保存模型
        print("保存模型...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"模型已保存到: {self.output_dir}")
        return trainer
    
    def generate_poetry(self, instruction, input_text, max_length=200, temperature=0.8, top_p=0.9):
        """使用 forward 方式生成诗歌"""
        prompt = f"指令: {instruction}\n要求: {input_text}\n诗词:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        self.model.eval()
        generated_tokens = inputs['input_ids'].clone()
        
        with torch.no_grad():
            for _ in range(max_length - inputs['input_ids'].shape[1]):
                # 前向传播
                outputs = self.model.forward(
                    input_ids=generated_tokens,
                    attention_mask=torch.ones_like(generated_tokens).to(self.device)
                )
                
                # 获取下一个token的logits
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # 应用top-p采样
                if temperature > 0:
                    # 计算累积概率
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 找到top-p阈值
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 移除低概率token
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成序列
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                
                # 如果生成了结束token，停止生成
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        poetry = generated_text.split("诗词:")[-1].strip()
        return poetry
    
    def batch_generate(self, prompts, **kwargs):
        """批量生成诗歌"""
        results = []
        for instruction, input_text in prompts:
            poetry = self.generate_poetry(instruction, input_text, **kwargs)
            results.append({
                "instruction": instruction,
                "input": input_text,
                "generated_poetry": poetry,
                "timestamp": datetime.now().isoformat()
            })
        return results
    
    def interactive_generation(self):
        """交互式诗歌生成"""
        print("\n=== 交互式诗歌生成 ===")
        print("输入 'quit' 退出")
        
        while True:
            try:
                instruction = input("\n请输入指令 (或 'quit' 退出): ").strip()
                if instruction.lower() == 'quit':
                    break
                
                input_text = input("请输入要求: ").strip()
                if not input_text:
                    print("要求不能为空")
                    continue
                
                print("\n正在生成诗歌...")
                poetry = self.generate_poetry(instruction, input_text)
                print(f"\n生成的诗歌:\n{poetry}")
                
            except KeyboardInterrupt:
                print("\n退出程序")
                break
            except Exception as e:
                print(f"生成失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="中文诗歌生成模型训练和推理")
    parser.add_argument("--mode", choices=["train", "generate", "interactive"], 
                       default="train", help="运行模式")
    parser.add_argument("--model-path", default="./poetry_model", 
                       help="模型路径")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max-samples", type=int, default=None, help="最大训练样本数（用于快速测试）")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # 训练模式
        trainer = PoetryTrainer(output_dir=args.model_path)
        trainer.load_and_explore_dataset(max_samples=args.max_samples)
        trainer.train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples
        )
        
    elif args.mode == "generate":
        # 生成模式
        if not os.path.exists(args.model_path):
            print(f"模型路径不存在: {args.model_path}")
            print("请先运行训练模式")
            return
        
        trainer = PoetryTrainer(output_dir=args.model_path)
        trainer.model = AutoModelForCausalLM.from_pretrained(args.model_path)
        trainer.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # 测试用例
        test_cases = [
            ("你是一个诗词创作的AI助手", "请你模仿李白的风格，创作一首关于月亮的七言绝句"),
            ("你是一个诗词创作的AI助手", "请你创作一首表达思乡之情的五言律诗"),
            ("你是一个诗词创作的AI助手", "请你写一首描写春天景色的词，词牌名不限"),
            ("你是一个诗词创作的AI助手", "请你创作一首关于友情的现代诗"),
        ]
        
        print("\n=== 诗歌生成测试 ===")
        results = trainer.batch_generate(test_cases)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- 测试用例 {i} ---")
            print(f"指令: {result['instruction']}")
            print(f"要求: {result['input']}")
            print("生成的诗词:")
            print(result['generated_poetry'])
        
        # 保存结果
        with open("generation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: generation_results.json")
        
    elif args.mode == "interactive":
        # 交互模式
        if not os.path.exists(args.model_path):
            print(f"模型路径不存在: {args.model_path}")
            print("请先运行训练模式")
            return
        
        trainer = PoetryTrainer(output_dir=args.model_path)
        trainer.model = AutoModelForCausalLM.from_pretrained(args.model_path)
        trainer.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        trainer.interactive_generation()

if __name__ == "__main__":
    main()
