import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Transformerに入力シーケンスの位置情報を付加するためのクラス。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # peを計算グラフに含めない定数として登録
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] (batch_first=Falseの場合)
               または [batch_size, seq_len, embedding_dim] (batch_first=Trueの場合)
        """
        # xの形状に合わせて位置エンコーディングを付加
        if x.dim() == 3 and self.pe.size(1) != x.size(1): # batch_first=True
             x = x + self.pe[:x.size(1)].transpose(0, 1)
        else: # batch_first=False
             x = x + self.pe[:x.size(0)]
        return self.dropout(x)