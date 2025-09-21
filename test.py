import torch
import torch.nn as nn
import torch.optim as optim


class Lora(nn.Module):
    def __init__(self,rank,alpha,in_dim,out_dim):
        self.rank=rank
        self.alpha=alpha
        tem=torch.tensor(rank**0.5)
        self.A=nn.Parameter(torch.randn(in_dim,rank)/tem)
        self.B=nn.Parameter(torch.zeros(rank,out_dim))
    def forward(self,x):

        out=x@self.A@self.B
        out=self.alpha/self.rank *out
        return out
