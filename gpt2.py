import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pdb
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# 加载数据集（只使用100条）
ds = load_dataset("Million/Chinese-Poems")['train'].select(range(100))
print(f"使用训练样本数: {len(ds)}")

# 数据预处理
def preprocess_function(examples):
    texts = [f"指令: {inst}\n要求: {inp}\n诗词: {out}" 
             for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output'])]
    
    tokenized = tokenizer(texts, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
    # 只对诗词部分计算损失
    labels = tokenized["input_ids"].clone()
    for i, text in enumerate(texts):
        poetry_start = text.find("诗词:") + 3
        prefix_tokens = tokenizer.encode(text[:poetry_start], add_special_tokens=False)
        labels[i, :len(prefix_tokens)+1] = -100
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# 应用预处理并创建DataLoader
tokenized_ds = ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)

class PoetryDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }

# 创建Dataset和DataLoader
poetry_dataset = PoetryDataset(tokenized_ds)
dataloader = DataLoader(poetry_dataset, batch_size=4, shuffle=True)

# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.to(device)

# 训练循环
# model.train()
# for epoch in tqdm(range(3)):
#     total_loss = 0
#     for batch in tqdm(dataloader):
#         inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#         labels = batch["labels"].to(device)
        
#         outputs = model(**inputs, labels=labels)
#         loss = outputs.loss
        
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         total_loss += loss.item()
        
#     print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


# 训练循环 - 底层实现详解
model.train()
for epoch in tqdm(range(3)):
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        print(f"\n=== Batch {batch_idx + 1} 详细过程 ===")
        
        # 1. 数据准备 - 底层张量操作
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        
        print(f"1. 输入数据形状:")
        print(f"   input_ids: {inputs['input_ids'].shape}")
        print(f"   attention_mask: {inputs['attention_mask'].shape}")
        print(f"   labels: {labels.shape}")
        
        # 2. 前向传播 - 手动计算过程
        print(f"\n2. 前向传播过程:")
        
        # 2.1 获取模型输出
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        loss = outputs.loss
        
        print(f"   模型输出logits形状: {logits.shape}")
        print(f"   自动计算的损失: {loss.item():.4f}")
        
        # 2.2 手动计算损失 - 底层实现
        print(f"\n3. 手动损失计算:")
        
        # 获取预测概率分布
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        labels_flat = labels.view(-1)  # [batch_size * seq_len]
        
        print(f"   展平后的logits形状: {logits_flat.shape}")
        print(f"   展平后的labels形状: {labels_flat.shape}")
        
        # 计算交叉熵损失
        log_probs = torch.log_softmax(logits_flat, dim=-1)
        
        # 创建掩码：忽略-100的位置
        mask = (labels_flat != -100)
        print(f"   有效位置数量: {mask.sum().item()}/{len(labels_flat)}")
        
        # 只对有效位置计算损失
        valid_log_probs = log_probs[mask]
        valid_labels = labels_flat[mask]
        
        pdb.set_trace()
        # 手动计算交叉熵
        manual_loss = -valid_log_probs[range(len(valid_labels)), valid_labels].mean()
        
        print(f"   手动计算的损失: {manual_loss.item():.4f}")
        print(f"   损失差异: {abs(loss.item() - manual_loss.item()):.6f}")
        
        # 3. 反向传播 - 梯度计算过程
        print(f"\n4. 反向传播过程:")
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 计算损失
        loss.backward()
        
        # 检查梯度
        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                total_grad_norm += grad_norm.item() ** 2
                param_count += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"   总梯度范数: {total_grad_norm:.4f}")
        print(f"   有梯度的参数数量: {param_count}")
        
        # 4. 参数更新 - 优化器内部过程
        print(f"\n5. 参数更新过程:")
        
        # 获取更新前的参数
        old_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                old_params[name] = param.data.clone()
        
        # 执行优化器步骤
        optimizer.step()
        
        # 计算参数变化
        param_changes = []
        for name, param in model.named_parameters():
            if param.requires_grad and name in old_params:
                change = (param.data - old_params[name]).norm().item()
                param_changes.append(change)
        
        avg_param_change = sum(param_changes) / len(param_changes) if param_changes else 0
        print(f"   平均参数变化: {avg_param_change:.6f}")
        
        total_loss += loss.item()
        
        # 只详细打印第一个batch
        if batch_idx == 0:
            print(f"\n6. 训练效果分析:")
            print(f"   当前batch损失: {loss.item():.4f}")
            print(f"   模型学会了什么:")
            print(f"   - 根据指令和要求生成诗词")
            print(f"   - 忽略前缀部分的重复模式")
            print(f"   - 专注于内容生成质量")
        
        if batch_idx >= 2:  # 只详细分析前3个batch
            break
    
    print(f"\nEpoch {epoch+1} 完成, 平均损失: {total_loss/len(dataloader):.4f}")


# 保存模型
model.save_pretrained("./poetry_model_small")
tokenizer.save_pretrained("./poetry_model_small")



# 诗歌生成函数 - 使用 forward 方式
def generate_poetry(instruction, input_text, max_length=200, temperature=0.8):
    """使用 forward 方式生成诗歌"""
    prompt = f"指令: {instruction}\n要求: {input_text}\n诗词:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    model.eval()
    generated_tokens = inputs['input_ids'].clone()
    
    with torch.no_grad():
        for _ in range(max_length - inputs['input_ids'].shape[1]):
            # 前向传播
            outputs = model.forward(
                input_ids=generated_tokens,
                attention_mask=torch.ones_like(generated_tokens).to(device)
            )
            
            # 获取下一个token的logits
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # 应用top-p采样
            if temperature > 0:
                # 计算累积概率
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 找到top-p阈值
                sorted_indices_to_remove = cumulative_probs > 0.9
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 移除低概率token
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            # 如果生成了结束token，停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    poetry = generated_text.split("诗词:")[-1].strip()
    return poetry

# 测试生成
print("\n=== 诗歌生成测试 ===")
test_cases = [
    ("你是一个诗词创作的AI助手", "请你创作一首关于春天的诗"),
    ("你是一个诗词创作的AI助手", "请你写一首表达思乡之情的诗"),
]

for i, (instruction, input_text) in enumerate(test_cases, 1):
    print(f"\n测试 {i}: {input_text}")
    try:
        poetry = generate_poetry(instruction, input_text)
        print(f"生成: {poetry}")
    except Exception as e:
        print(f"生成失败: {e}")

print("\n训练完成！")