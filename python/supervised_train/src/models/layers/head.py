import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Dict

class _MultiScaleHead(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.heads_list = nn.ModuleList() 
        self.head_keys = [] 
        
        for key, in_ch in cfg.in_channels.items():
            mid_features = int(in_ch * cfg.get('mid_features_ratio', 0.5))
            
            self.heads_list.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(in_ch, mid_features),
                    nn.SiLU(inplace=True),
                    nn.Linear(mid_features, cfg.out_features)
                )
            )
            self.head_keys.append(key)
            
        self.steer_activation = nn.Tanh()

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        for i, head_module in enumerate(self.heads_list): # ここで i と head_module を同時に取得
            key = self.head_keys[i] # i を使って対応するキーを取得
            x = features[key] 

            # head_module を直接呼び出す
            pred = head_module(x) 
            
            steer_out = self.steer_activation(pred[:, 0])
            speed_out = pred[:, 1]
            
            outputs[key] = torch.stack([steer_out, speed_out], dim=1)
            
        return outputs