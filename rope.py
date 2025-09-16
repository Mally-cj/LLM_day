import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 初始化频率参数
        theta = 1.0 / (10000 ** torch.arange(0, dim, 2, dtype=torch.float) / dim)
        self.register_buffer('theta', theta)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None):
        batch, seq_len, dim = x.shape
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        # 计算每个位置的角度
        angles = positions.unsqueeze(-1) * self.theta.unsqueeze(0)  # seq_len, dim//2
        
        cos = torch.cos(angles)  # seq_len, dim//2
        sin = torch.sin(angles)  # seq_len, dim//2
        
        x1, x2 = x[..., 0::2], x[..., 1::2] # 每两个维度为一组
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)


if __name__ == "__main__":
    dim = 64
    max_seq_len = 128
    batch_size = 32
    seq_len = 50
    
    rope = RotaryPositionEmbedding(dim)
    x = torch.randn(batch_size, seq_len, dim)  # (batch_size, seq_len, dim)
    
    x_rotated = rope(x)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {x_rotated.shape}")



