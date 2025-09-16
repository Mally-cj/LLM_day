import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        # 可学习的缩放参数，初始化为1
        self.gamma = nn.Parameter(torch.ones(dim))
        # 可学习的偏移参数，初始化为0
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 使用均值和标准差对输入进行归一化
        x_norm = (x - mean) / (std + self.eps)
        x = x * self.gamma + self.beta

        return x


x = torch.randn((100, 64, 512))
norm = LayerNorm(512)
x = norm(x)
print(x.shape)