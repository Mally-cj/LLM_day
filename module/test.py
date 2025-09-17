
import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.linear_q = nn.Linear(hidden_dim, hidden_dim)