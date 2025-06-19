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
        
        self.downsample_conv = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2)
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
        
        x = self.downsample_conv(x)
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
    各ステージの冒頭で一度だけダウンサンプリングを行う。
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, partition_size: int, drop_path_rates: list, dim_head: int, norm_layer: nn.Module):
        super().__init__()
        
        # ステージの冒頭でダウンサンプリング層を定義
        # Stage1dに入力された時点でのin_channelsとout_channelsを使用
        self.downsample = DownsampleLayer1d(in_channels, out_channels)
            
        blocks = []
        for i in range(num_blocks):
            # Stage1dのダウンサンプリング後のチャンネル数(out_channels)をブロックの入力とする
            blocks.append(PartitionAttention1d(out_channels, 'window', partition_size, dim_head=dim_head, drop_path=drop_path_rates[i*2]))
            blocks.append(PartitionAttention1d(out_channels, 'grid', partition_size, dim_head=dim_head, drop_path=drop_path_rates[i*2+1]))
        
        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ステージの最初にダウンサンプリングを実行
        x = self.downsample(x) 
        # その後、全てのブロックを適用
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
        
        out_indices_list = list(cfg.out_indices)
        out_keys_list = list(cfg.out_keys)
        
        num_stages = len(cfg.dims)
        # LidarStemの出力は target_seq_len / 2 になるように変更
        # Stemのout_channelsは最初のステージのin_chと同じになるように調整
        self.stem = LidarStem(cfg.input_channels, cfg.dims[0], cfg.target_seq_len)

        # partition_sizesの計算も、stemで1回ダウンサンプリングされていることを考慮
        # LidarStemの出力系列長は target_seq_len / 2
        initial_seq_len_after_stem = cfg.target_seq_len // 2
        # 各ステージでの系列長は (initial_seq_len_after_stem / (2^i)) となる
        partition_sizes = [max(initial_seq_len_after_stem // (2**i) // cfg.partition_ratio, 1) for i in range(num_stages)]
        
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.num_blocks) * 2)]
        dim_head = cfg.get('dim_head', 32)
        
        self.stages = nn.ModuleList()
        block_cursor = 0
        for i in range(num_stages):
            # 最初のステージ (i=0) のin_chはLidarStemのout_channels (cfg.dims[0])
            # それ以降のステージ (i>0) のin_chは前のステージのout_ch (cfg.dims[i-1])
            in_ch = cfg.dims[i] if i == 0 else cfg.dims[i-1] # 修正: 最初のin_chはcfg.dims[0]
            out_ch = cfg.dims[i]
            # downsampleは常にTrueなので、引数から削除
            dpr_slice = dpr[block_cursor : block_cursor + cfg.num_blocks[i] * 2]
            
            norm_layer: nn.Module
            if i in out_indices_list:
                norm_layer = nn.LayerNorm(out_ch)
            else:
                norm_layer = nn.Identity()
            
            self.stages.append(
                Stage1d(in_ch, out_ch, cfg.num_blocks[i], partition_sizes[i], dpr_slice, dim_head=dim_head, norm_layer=norm_layer) # downsample引数を削除
            )
            block_cursor += cfg.num_blocks[i] * 2

        self.out_indices = out_indices_list
        self.idx_to_key_map: Dict[int, str] = {idx: key for idx, key in zip(out_indices_list, out_keys_list)}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        
        features: Dict[str, torch.Tensor] = {}
        # C0はstemの出力となるため、ループに入る前にfeaturesに追加
        # ただし、out_indicesに0が含まれていなければ追加しない
        if 0 in self.out_indices:
            key_for_stem = self.idx_to_key_map[0]
            features[key_for_stem] = x

        for i, stage in enumerate(self.stages):
            
            x = stage(x) # 各ステージはダウンサンプリングを行う

            if (i + 1) in self.out_indices: # Stage0の出力はC1、Stage1の出力はC2など
                key = self.idx_to_key_map[i + 1] # out_indices_listのインデックスとステージ番号のずれを調整
                features[key] = x
        
        return features