import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        # 维度必须能被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim must be multiple of num_heads"
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        # 定义线性变换矩阵
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self._norm_fact = math.sqrt(hidden_dim // num_heads)
 
    def forward(self, x):
        # x: tensor of shape (batch, seq_len, hidden_dim)
        bsz, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim
 
        nh = self.num_heads
        d = self.hidden_dim // nh  # dim of each head
 
        q = self.linear_q(x).reshape(bsz, seq_len, nh, d).transpose(1, 2)  # (bsz, nh, seq_len, d)
        k = self.linear_k(x).reshape(bsz, seq_len, nh, d).transpose(1, 2)  # (bsz, nh, seq_len, d)
        v = self.linear_v(x).reshape(bsz, seq_len, nh, d).transpose(1, 2)  # (bsz, nh, seq_len, d)

        # b nh s d * b nh d s = b nh s s
        att = torch.matmul(q, k.transpose(2, 3))
        # norm
        att = att / self._norm_fact
        # softmax
        att = torch.softmax(att, dim=-1)  # bsz, nh, s, s
        # dropout
        att = F.dropout(att, p=0.5, training=self.training)
        # bsz, nh, s, s * bsz, nh, s, d = bsz, nh, s, d
        out = torch.matmul(att, v)
        # bsz, nh, s, d ---> bsz, s, nh, d ---> bsz, s, nh*d=hidden_dim
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)

        return self.out(out)


# mha = MultiHeadAttention(64)
# x = torch.randn((256, 1024, 64))
# x = mha(x)
# print(x.shape)
