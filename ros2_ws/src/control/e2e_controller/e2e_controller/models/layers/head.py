import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Dict

class _MultiScaleHead(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.heads_list = nn.ModuleList() 
        self.head_keys = [] 
        
        # ストライド情報を保持
        self.strides_map = cfg.strides_map
        if not self.strides_map:
            raise ValueError("strides_map must be provided in head config for multi-scale decoding.")
        
        for key, in_ch in cfg.in_channels.items():
            mid_features = int(in_ch * cfg.get('mid_features_ratio', 0.5))
            
            self.heads_list.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1), # (B, L, C) -> (B, 1, C)
                    nn.Flatten(),             # (B, C)
                    nn.Linear(in_ch, mid_features),
                    nn.SiLU(inplace=True),
                    nn.Linear(mid_features, cfg.out_features) # out_features は SteerとSpeedの2つを想定
                )
            )
            self.head_keys.append(key)
            
        self.steer_activation = nn.Tanh()

    def decode_output(self, raw_pred: torch.Tensor, stride: int) -> Dict[str, torch.Tensor]:
        steer_raw = raw_pred[:, 0]
        speed_raw = raw_pred[:, 1]

        steer_out = self.steer_activation(steer_raw * stride)
        
        steer_out = self.steer_activation(steer_raw)
        speed_out = F.relu(speed_raw)
        
        return {
            'steer': steer_out,
            'speed': speed_out
        }

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: # 戻り値の型ヒントも修正
        outputs = {}
        
        # features は PAFPN1d の出力であり、各キー ('C1', 'C2' など) に対応する
        # 特徴マップを含むディクショナリであると想定
        
        for i, head_module in enumerate(self.heads_list):
            key = self.head_keys[i] # 例: 'C1', 'C2', ...
            
            # PAFPN1dがそのキーの特徴量を返していることを前提とする
            if key not in features:
                raise KeyError(f"Feature key '{key}' expected by _MultiScaleHead but not found in FPN outputs.")
            
            x = features[key] 

            # 各ヘッドで予測を生成 (B, out_features)
            raw_pred = head_module(x) 
            
            # ストライドを取得
            stride = self.strides_map.get(key)
            if stride is None:
                raise ValueError(f"Stride not found for key: {key}. Check cfg.strides_map.")

            outputs[key] = raw_pred
            
            
        return outputs