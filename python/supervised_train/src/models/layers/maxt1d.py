import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from omegaconf import DictConfig

from .blocks import PartitionAttention1d

class LidarStem(nn.Module):
    """
    モデルの入力部分。1D-Convで初期特徴量を抽出する。
    """
    def __init__(self, in_channels: int, out_channels: int, target_seq_len: int):
        super().__init__()
        
        self.target_seq_len = target_seq_len
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力シーケンス長を指定の長さに合わせる (Padding or Truncating)
        current_len = x.shape[-1]
        delta = self.target_seq_len - current_len
        
        if delta > 0:
            x = F.pad(x, (0, delta))
        elif delta < 0:
            x = x[..., :self.target_seq_len]
        
        # 畳み込みブロックを適用
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Transformerブロックで扱いやすいように次元を入れ替える (B, C, L) -> (B, L, C)
        return x.permute(0, 2, 1)


class DownsampleLayer1d(nn.Module):
    """
    LayerNormと1D-Convによるダウンサンプリング層。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, C)
        x = self.norm(x)
        # (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # (B, L', C')
        x = x.permute(0, 2, 1)
        return x


class Stage1d(nn.Module):
    """
    正規化層を内部に持つ、自己完結したステージ。
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, partition_size: int, drop_path_rates: list, downsample: bool, dim_head: int, norm_layer: nn.Module):
        super().__init__()
        
        if downsample:
            self.downsample = DownsampleLayer1d(in_channels, out_channels)
        else:
            self.downsample = nn.Identity()
            assert in_channels == out_channels
            
        blocks = []
        for i in range(num_blocks):
            blocks.append(PartitionAttention1d(out_channels, 'window', partition_size, dim_head=dim_head, drop_path=drop_path_rates[i*2]))
            blocks.append(PartitionAttention1d(out_channels, 'grid', partition_size, dim_head=dim_head, drop_path=drop_path_rates[i*2+1]))
        
        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer  # 受け取った正規化層を保持
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        x = self.norm(x)  
        return x
    

class MaxT1d(nn.Module):
    """
    MaxT1dバックボーン。
    常にマルチスケールの特徴量マップを出力する回帰タスク専用モデル。
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        # JITが理解できるPythonネイティブの型に変換
        out_indices_list = list(cfg.out_indices)
        out_keys_list = list(cfg.out_keys)
        
        num_stages = len(cfg.dims)
        self.stem = LidarStem(cfg.input_channels, cfg.dims[0], cfg.target_seq_len)
        partition_sizes = [max(cfg.target_seq_len // (2**i) // cfg.partition_ratio, 1) for i in range(num_stages)]
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.num_blocks) * 2)]
        dim_head = cfg.get('dim_head', 32)
        
        self.stages = nn.ModuleList()
        block_cursor = 0
        for i in range(num_stages):
            in_ch = cfg.dims[i-1] if i > 0 else cfg.dims[0]
            out_ch = cfg.dims[i]
            downsample = (i > 0)
            dpr_slice = dpr[block_cursor : block_cursor + cfg.num_blocks[i] * 2]
            
            # 出力するステージではLayerNormを、それ以外ではIdentityを適用
            norm_layer: nn.Module
            if i in out_indices_list:
                norm_layer = nn.LayerNorm(out_ch)
            else:
                norm_layer = nn.Identity()
            
            self.stages.append(
                Stage1d(in_ch, out_ch, cfg.num_blocks[i], partition_sizes[i], dpr_slice, downsample, dim_head=dim_head, norm_layer=norm_layer)
            )
            block_cursor += cfg.num_blocks[i] * 2

        # ▼▼▼ 分類モード用の属性を全て削除 ▼▼▼
        # self.norm, self.pool, self.head は不要

        # ▼▼▼ 特徴量抽出モード用の属性のみを保持 ▼▼▼
        self.out_indices = out_indices_list
        self.idx_to_key_map: Dict[int, str] = {idx: key for idx, key in zip(out_indices_list, out_keys_list)}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # ▼▼▼ forwardメソッドからif/elseを完全に削除 ▼▼▼
        x = self.stem(x)
        
        features: Dict[str, torch.Tensor] = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                key = self.idx_to_key_map[i]
                features[key] = x
        return features