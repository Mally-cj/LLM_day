import torch
import torch.nn as nn
import torch.nn.functional as F
from mha import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        """
        d_model: 输入特征维度
        nhead: 多头注意力机制中的头数
        dim_feedforward: 前馈网络中间层维度
        dropout: Dropout比例
        activation: 激活函数类型 ('relu' 或 'gelu')
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # 自注意力机制
        self.self_attn = MultiHeadAttention(d_model, num_heads=nhead)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 设置激活函数
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # att
        src2 = self.self_attn(src)
        # 添加残差连接并进行层归一化，这里是post-norm
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # 前馈神经网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 添加残差连接并进行层归一化
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise ValueError(f"未知的激活函数 {activation}")


if __name__ == "__main__":
    d_model = 512
    nhead = 8
    batch_size = 32
    seq_len = 10

    src = torch.rand(batch_size, seq_len, d_model)

    encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    output = encoder_layer(src)
    print(output.shape)