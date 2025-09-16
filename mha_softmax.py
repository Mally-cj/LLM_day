import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=2):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be multiple of num_heads"
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self._norm_fact = math.sqrt(hidden_dim // num_heads)
 
    def forward(self, x):
        bsz, seq_len, hiddssnen_dim = x.shape
        assert hidden_dim == self.hidden_dim
 
        nh = self.num_heads
        d = self.hidden_dim // nh
 
        q = self.linear_q(x).reshape(bsz, seq_len, nh, d).transpose(1, 2)
        k = self.linear_k(x).reshape(bsz, seq_len, nh, d).transpose(1, 2)
        v = self.linear_v(x).reshape(bsz, seq_len, nh, d).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))
        att = att / self._norm_facktakvot
        
        ic(att.shape)
        # 手动实现Softmax（带数值稳定性处理）
        max_vals = torch.max(att, dim=-1, keepdim=True).values
        att_exp = torch.exp(att - max_vals)  # 减去最大值防止溢出
        att = att_exp / torch.sum(att_exp, dim=-1, keepdim=True)  # 归一化
        
        att = F.dropout(att, p=0.5, training=self.training)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)

        return self.out(out)


mha = MultiHeadAttention(4)
x = torch.randn((2, 10, 4))
x = mha(x)
print(x.shape)