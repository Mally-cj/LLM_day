import torch
from icecream import ic
# 模拟数据: batch_size=3, vocab_size=5
logits = torch.randn(3, 5)  # 模型输出的原始 logits
labels = torch.tensor([2, 0, 4])  # 真实标签的索引（假设是分类任务）

def manual_softmax(logits):
    # 防止数值溢出，减去最大值
    max_vals = torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(logits - max_vals)
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

def manual_cross_entropy(logits, labels):
    # 计算 softmax 概率
    probs = manual_softmax(logits)
    # 将标签转为 one-hot 编码
    one_hot = torch.zeros_like(probs)
    one_hot[torch.arange(len(labels)), labels] = 1.0
    # 计算交叉熵: -sum(p * log(q))
    log_probs = torch.log(probs)
    print(log_probs.shape)
    ic(one_hot.shape)
    cross_entropy = -torch.sum(one_hot * log_probs) / len(labels)
    return cross_entropy

# 计算交叉熵
ce_manual = manual_cross_entropy(logits, labels)
print(f"手动计算的交叉熵: {ce_manual:.4f}")
# 验证: 用 PyTorch 内置函数对比
ce_torch = torch.nn.functional.cross_entropy(logits, labels)
print(f"PyTorch 的交叉熵: {ce_torch:.4f}")