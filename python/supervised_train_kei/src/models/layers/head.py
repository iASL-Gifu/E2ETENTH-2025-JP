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
        
        self.strides_map = cfg.strides_map
        if not self.strides_map:
            raise ValueError("strides_map must be provided in head config for multi-scale decoding.")
        
        # 出力次元は常に out_features (mu のみ)
        output_dim = cfg.out_features
        
        for key, in_ch in cfg.in_channels.items():
            mid_features = int(in_ch * cfg.get('mid_features_ratio', 0.5))
            
            self.heads_list.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1), 
                    nn.Flatten(),             
                    nn.Linear(in_ch, mid_features),
                    nn.SiLU(inplace=True),
                    nn.Linear(mid_features, output_dim) # 固定された出力次元を使用
                )
            )
            self.head_keys.append(key)
            
        self.steer_activation = nn.Tanh()
        self.out_features = cfg.out_features 

    def decode_output(self, raw_pred: torch.Tensor) -> torch.Tensor:
        """
        steerとspeedの予測値を結合した単一のテンソル (B, 2) を返すように修正。
        """
        mu_pred = raw_pred

        steer_mu = mu_pred[:, 0].unsqueeze(1)
        speed_mu = mu_pred[:, 1].unsqueeze(1)
        
        steer_out = self.steer_activation(steer_mu) 
        speed_out = F.relu(speed_mu)
        
        # 辞書ではなく、結合したテンソルを返す
        return torch.cat([steer_out, speed_out], dim=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        各キーに対応する予測テンソルを格納した辞書を返すように修正。
        """
        outputs = {}
        
        for i, head_module in enumerate(self.heads_list):
            key = self.head_keys[i] 
            
            if key not in features:
                raise KeyError(f"Feature key '{key}' expected by _MultiScaleHead but not found in FPN outputs.")
            
            x = features[key] 
            raw_pred = head_module(x) 
            
            # decoded_output は (B, 2) のテンソル
            decoded_output = self.decode_output(raw_pred)
            outputs[key] = decoded_output
            
        return outputs