import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
from beam_search import BeamSearch


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class MaskedMultiHeadAttention(nn.Module):
    """带掩码的多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = query.size()
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        
        # 重塑并输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.w_o(context)


class CrossAttention(nn.Module):
    """交叉注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = query.size()
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        
        # 重塑并输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.w_o(context)


class DecoderLayer(nn.Module):
    """Transformer Decoder层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # 自注意力（带掩码）
        self.self_attn = MaskedMultiHeadAttention(d_model, num_heads, dropout)
        
        # 交叉注意力
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """完整的Transformer Decoder"""
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_len: int = 5000, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Decoder层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """创建因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: 目标序列 [batch_size, tgt_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            tgt_mask: 目标序列掩码
            src_mask: 源序列掩码
        """
        batch_size, tgt_len = tgt.size()
        
        # 词嵌入
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 创建因果掩码
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt_len)
        
        # 通过Decoder层
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output


class LanguageModel(nn.Module):
    """语言模型，用于训练和推理"""
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 简化的编码器（实际应用中可能是完整的编码器）
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.Linear(d_model, d_model)
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
        """
        # 编码器输出
        encoder_output = self.encoder(src)
        
        # Decoder输出
        output = self.decoder(tgt, encoder_output)
        
        return output
    
    def generate(self, src: torch.Tensor, max_len: int = 50, 
                 temperature: float = 1.0, top_k: int = None) -> List[int]:
        """生成文本"""
        self.eval()
        with torch.no_grad():
            # 编码源序列
            encoder_output = self.encoder(src)
            
            # 初始化目标序列（使用特殊token，这里用0）
            tgt = torch.tensor([[0]], device=src.device)  # [1, 1]
            
            generated = []
            
            for _ in range(max_len):
                # 前向传播
                output = self.decoder(tgt, encoder_output)
                
                # 获取最后一个时间步的logits
                next_token_logits = output[0, -1, :] / temperature
                
                # Top-k采样
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 检查结束条件（这里简化处理）
                if next_token.item() == 0:  # 假设0是结束token
                    break
                
                generated.append(next_token.item())
                
                # 更新目标序列
                tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            
            return generated
    
    def predict_next_word(self, sequence: List[int]) -> np.ndarray:
        """为beam search提供下一个词的概率分布"""
        self.eval()
        with torch.no_grad():
            # 转换为tensor
            src = torch.tensor([sequence], dtype=torch.long)
            tgt = torch.tensor([sequence], dtype=torch.long)
            
            # 前向传播
            output = self.forward(src, tgt)
            
            # 获取最后一个时间步的logits并转换为概率
            next_token_logits = output[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            
            return probs.cpu().numpy()


def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """创建填充掩码"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)


def train_model(model: LanguageModel, train_data: List[Tuple[List[int], List[int]]],
                epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充token
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # 简单的批处理
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            
            # 准备批次数据
            src_batch = []
            tgt_batch = []
            
            for src, tgt in batch:
                src_batch.append(src)
                tgt_batch.append(tgt[:-1])  # 输入序列（去掉最后一个token）
            
            # 填充到相同长度
            max_src_len = max(len(s) for s in src_batch)
            max_tgt_len = max(len(t) for t in tgt_batch)
            
            src_tensor = torch.zeros(len(batch), max_src_len, dtype=torch.long)
            tgt_tensor = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)
            
            for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
                src_tensor[i, :len(src)] = torch.tensor(src)
                tgt_tensor[i, :len(tgt)] = torch.tensor(tgt)
            
            # 移动到设备
            src_tensor = src_tensor.to(device)
            tgt_tensor = tgt_tensor.to(device)
            
            # 前向传播
            output = model(src_tensor, tgt_tensor)
            
            # 计算损失
            target = torch.zeros_like(tgt_tensor)
            for i, (_, tgt) in enumerate(batch):
                target[i, :len(tgt[1:])] = torch.tensor(tgt[1:])  # 目标序列（去掉第一个token）
            
            target = target.to(device)
            loss = criterion(output.view(-1, model.vocab_size), target.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # 示例：创建和训练模型
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # 创建模型
    model = LanguageModel(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # 生成示例训练数据
    train_data = []
    for _ in range(100):
        # 随机生成源序列和目标序列
        src_len = np.random.randint(5, 20)
        tgt_len = np.random.randint(5, 20)
        
        src = np.random.randint(1, vocab_size, src_len).tolist()
        tgt = np.random.randint(1, vocab_size, tgt_len).tolist()
        
        train_data.append((src, tgt))
    
    print("开始训练...")
    train_model(model, train_data, epochs=5, batch_size=8)
    
    # 测试生成
    print("\n测试生成...")
    src = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    generated = model.generate(src, max_len=20)
    print(f"生成的序列: {generated}")
    
    # 测试beam search
    print("\n测试Beam Search...")
    beam_search = BeamSearch(model, beam_width=3)
    start_sequence = [1, 2, 3]
    best_sequence, best_score = beam_search.search(start_sequence, max_length=10)
    print(f"Beam Search结果: {best_sequence}")
    print(f"得分: {best_score:.4f}")
