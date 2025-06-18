import torch
import torch.nn as nn
from typing import Optional

# ===================================================================
# Helper Modules
# ===================================================================

class DropPath(nn.Module):
    """
    Stochastic Depthの実装。
    指定された確率で入力をゼロにする（Residual Connectionのドロップアウト）。
    """
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # (B, 1, 1, ...) のような形状にし、ブロードキャスト可能にする
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0か1のバイナリマスクにする
        
        # スケールを調整して期待値を保つ
        return x.div(keep_prob) * random_tensor


# ===================================================================
# Convolution Blocks
# ===================================================================

class BaseConv1d(nn.Module):
    """
    基本的な Conv1d + BatchNorm1d + Activation のブロック。
    """
    def __init__(self, in_channels: int, out_channels: int, ksize: int, stride: int, groups: int = 1, act: str = "silu"):
        super().__init__()
        
        pad = (ksize - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, ksize, stride, padding=pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv1d(nn.Module):
    """
    Depthwise Separable Convolution Block (1D).
    """
    def __init__(self, in_channels: int, out_channels: int, ksize: int, stride: int = 1, act: str = "silu"):
        super().__init__()
        # Depthwise Convolution
        self.dconv = BaseConv1d(in_channels, in_channels, ksize, stride, groups=in_channels, act=act)
        # Pointwise Convolution
        self.pconv = BaseConv1d(in_channels, out_channels, 1, 1, groups=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dconv(x)
        x = self.pconv(x)
        return x


# ===================================================================
# CSP / Bottleneck Blocks
# ===================================================================

class Bottleneck1d(nn.Module):
    """
    CSPNetで使われるBottleneckブロックの1D版。
    """
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, expansion: float = 0.5, depthwise: bool = False, act: str = "silu"):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv1d if depthwise else BaseConv1d
        
        self.conv1 = BaseConv1d(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        
        if self.use_add:
            return y + x
        else:
            return y


class CSPLayer1d(nn.Module):
    """
    CSP (Cross Stage Partial) レイヤーの1D版。
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True, expansion: float = 0.5, depthwise: bool = False, act: str = "silu"):
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = BaseConv1d(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = BaseConv1d(in_channels, hidden_channels, 1, 1, act=act)
        
        self.bottlenecks = nn.Sequential(
            *[Bottleneck1d(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        )
        
        self.conv3 = BaseConv1d(2 * hidden_channels, out_channels, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        
        x_2 = self.conv2(x)
        
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        return x


# ===================================================================
# Attention Blocks
# ===================================================================

class SelfAttention1d(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, bias: bool = True):
        super().__init__()
        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        # Q, K, Vを計算し、ヘッドごとに分割
        qkv = self.qkv(x).view(B, L, self.num_heads, self.dim_head * 3)
        q, k, v = qkv.transpose(1, 2).chunk(3, dim=3)
        
        # Attentionスコアを計算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Attentionを適用し、ヘッドを結合
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        
        return out



class PartitionAttention1d(nn.Module):
    def __init__(self, dim: int, partition_type: str, partition_size: int, dim_head: int = 32, mlp_ratio: float = 4.0, drop_path: float = 0.):
        super().__init__()
        
        self.partition_type = partition_type
        self.partition_size = partition_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention1d(dim=dim, dim_head=dim_head)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim)
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # ▼▼▼▼▼ このメソッドを修正 ▼▼▼▼▼
    def _partition_attn(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        
        if self.partition_type == 'window':
            # ifブロック内で直接関数を呼び出す
            partitioned = window_partition_1d(x, self.partition_size)
            partitioned_attn = self.self_attn(partitioned)
            reversed_x = window_reverse_1d(partitioned_attn, self.partition_size, seq_len)
        else: # 'grid'
            # elseブロック内で直接関数を呼び出す
            partitioned = grid_partition_1d(x, self.partition_size)
            partitioned_attn = self.self_attn(partitioned)
            reversed_x = grid_reverse_1d(partitioned_attn, self.partition_size, seq_len)
            
        return reversed_x
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Residual
        x = x + self.drop_path1(self._partition_attn(self.norm1(x)))
        # MLP + Residual
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ===================================================================
# Partition/Reverse Functions
# ===================================================================
@torch.jit.script
def window_partition_1d(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, L, C = x.shape
    assert L % window_size == 0, f"L ({L}) must be divisible by window_size ({window_size})"
    
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows

@torch.jit.script
def window_reverse_1d(windows: torch.Tensor, window_size: int, seq_len: int) -> torch.Tensor:
    B_times_num_windows, _, C = windows.shape
    num_windows = seq_len // window_size
    B = B_times_num_windows // num_windows
    
    x = windows.view(B, num_windows, window_size, C)
    x = x.contiguous().view(B, seq_len, C)
    return x

@torch.jit.script
def grid_partition_1d(x: torch.Tensor, grid_size: int) -> torch.Tensor:
    B, L, C = x.shape
    assert L % grid_size == 0, f"L ({L}) must be divisible by grid_size ({grid_size})"

    x = x.view(B, grid_size, L // grid_size, C)
    # (B, grid_size, tokens_per_grid, C) -> (B, tokens_per_grid, grid_size, C)
    x = x.permute(0, 2, 1, 3).contiguous()
    grids = x.view(-1, grid_size, C)
    return grids

@torch.jit.script
def grid_reverse_1d(grids: torch.Tensor, grid_size: int, seq_len: int) -> torch.Tensor:
    _B_times_tokens, _, C = grids.shape
    tokens_per_grid = seq_len // grid_size
    B = _B_times_tokens // tokens_per_grid
    
    x = grids.view(B, tokens_per_grid, grid_size, C)
    # (B, tokens_per_grid, grid_size, C) -> (B, grid_size, tokens_per_grid, C)
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(B, seq_len, C)
    return x