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
        
        # 新しいフラグをインスタンス変数として保存
        self.predict_uncertainty = cfg.get('predict_uncertainty', True) # デフォルト値を設定
        
        # 出力次元を決定
        # 信頼性予測が有効なら out_features * 2 (mu + log_sigma^2)
        # 無効なら out_features (mu のみ)
        output_dim = cfg.out_features * 2 if self.predict_uncertainty else cfg.out_features
        
        for key, in_ch in cfg.in_channels.items():
            mid_features = int(in_ch * cfg.get('mid_features_ratio', 0.5))
            
            self.heads_list.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1), 
                    nn.Flatten(),             
                    nn.Linear(in_ch, mid_features),
                    nn.SiLU(inplace=True),
                    nn.Linear(mid_features, output_dim) # output_dim を使用
                )
            )
            self.head_keys.append(key)
            
        self.steer_activation = nn.Tanh()
        self.out_features = cfg.out_features 

    # decode_output の引数を修正
    def decode_output(self, raw_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.predict_uncertainty:
            mu_pred = raw_pred[:, :self.out_features]
            log_sigma_squared_pred = raw_pred[:, self.out_features:]
        else:
            mu_pred = raw_pred
            log_sigma_squared_pred = None

        # ここを修正: unsqueeze(1) を追加して (B,) -> (B, 1) にする
        steer_mu = mu_pred[:, 0].unsqueeze(1) # (batch_size, 1)
        speed_mu = mu_pred[:, 1].unsqueeze(1) # (batch_size, 1)
        
        steer_out = self.steer_activation(steer_mu) 
        speed_out = F.relu(speed_mu)
        
        output_dict = {
            'steer_mu': steer_out,
            'speed_mu': speed_out
        }
        
        if self.predict_uncertainty:
            output_dict['steer_log_sigma_squared'] = log_sigma_squared_pred[:, 0].unsqueeze(1) # (batch_size, 1)
            output_dict['speed_log_sigma_squared'] = log_sigma_squared_pred[:, 1].unsqueeze(1) # (batch_size, 1)
            
        return output_dict


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        outputs = {}
        
        for i, head_module in enumerate(self.heads_list):
            key = self.head_keys[i] 
            if key not in features:
                raise KeyError(f"Feature key '{key}' expected by _MultiScaleHead but not found in FPN outputs.")
            
            x = features[key] 

            raw_pred = head_module(x) 
            
            decoded_output = self.decode_output(raw_pred)
            outputs[key] = decoded_output
            
        return outputs