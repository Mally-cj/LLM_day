import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为1
        self.gamma = nn.Parameter(torch.ones(dim))
        # 可学习的偏移参数，初始化为0
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # norm
        x_norm = x / rms
        return self.gamma * x_norm + self.beta


x = torch.randn((100, 64, 512))
norm = RMSNorm(512)
x = norm(x)
print(x.shape)