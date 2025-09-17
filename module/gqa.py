import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, q_heads=8, key_value_heads=2):
        super(GroupedQueryAttention, self).__init__()
        # 维度必须能被num_heads整除
        assert hidden_dim % q_heads == 0, "hidden_dim must be multiple of q_heads"
        self.q_heads = q_heads
        self.key_value_heads = key_value_heads
        self.hidden_dim = hidden_dim
        self.num_groups = q_heads // key_value_heads
        self.head_dim = hidden_dim // q_heads
        # 定义线性变换矩阵
        self.linear_q = nn.Linear(hidden_dim, q_heads * self.head_dim)
        self.linear_k = nn.Linear(hidden_dim, key_value_heads * self.head_dim)
        self.linear_v = nn.Linear(hidden_dim, key_value_heads * self.head_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self._norm_fact = math.sqrt(self.head_dim)
 
    def forward(self, x):
        # x: tensor of shape (batch, seq_len, hidden_dim)
        bsz, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim
 
        d = self.head_dim  # dim of each head
 
        q = self.linear_q(x).reshape(bsz, seq_len, self.q_heads, d)  # (bsz, seq_len, q_heads, d)
        k = self.linear_k(x).reshape(bsz, seq_len, self.key_value_heads, d).transpose(2, 3)  # (bsz, seq_len, d, kv_heads)
        v = self.linear_v(x).reshape(bsz, seq_len, self.key_value_heads, d)  # (bsz, seq_len, kv_heads, d)

        # 对query进行分组
        # (bsz, seq_len, q_heads, d) ---> (bsz, seq_len, groups, key_value_heads, d)
        q = q.view(bsz, -1, self.num_groups, self.key_value_heads, self.head_dim)

        # (bsz, seq_len, groups, kv_heads, d) * (bsz, seq_len, d, kv_heads) = (bsz, seq_len, groups, kv_heads, kv_heads)
        att = torch.einsum('bsgkd,bsdt->bsgkt', q, k)
        # norm
        att = att / self._norm_fact
        # softmax
        att = torch.softmax(att, dim=-1)  #
        # dropout
        att = F.dropout(att, p=0.5, training=self.training)
        # (bsz, seq_len, groups, kv_heads, kv_heads) * (bsz, seq_len, kv_heads, d) = (bsz, seq_len, groups, kv_heads, kv_heads)
        out = torch.einsum('bsgtk,bskd->bsgtd', att, v)
        # bsgkd ---> bsz, s, nh, d ---> bsz, s, nh*d=hidden_dim
        out = out.view(bsz, seq_len, self.q_heads, d).view(bsz, seq_len, self.hidden_dim)

        return self.out(out)


gqa = GroupedQueryAttention(64, q_heads=8, key_value_heads=2)
x = torch.randn((256, 1024, 64))
x = gqa(x)
print(x.shape)
