import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from typing import Union, Dict

from .layers.maxt1d import MaxT1d
from .layers.fpn1d import PAFPN1d
from .layers.head import _MultiScaleHead

def get_model_cfg(
    model_size: str, 
    backbone_stages: int = 3, 
    fpn_stages: int = 3
) -> DictConfig:
    """ 
    モデルのサイズとステージ数から設定を生成する。
    """
    if fpn_stages > backbone_stages:
        raise ValueError("fpn_stages cannot be greater than backbone_stages")

    # サイズ別のパラメータ定義
    if model_size == 'tiny':
        dims = [32, 64, 128, 256]; num_blocks = [2, 2, 5, 2]; neck_depth = 0.33
        dim_head = 32
    elif model_size == 'small':
        dims = [48, 96, 192, 384]; num_blocks = [2, 2, 5, 2]; neck_depth = 0.67
        dim_head = 24
    elif model_size == 'base':
        dims = [64, 128, 256, 512]; num_blocks = [2, 2, 5, 2]; neck_depth = 1.0
        dim_head = 32
    else: 
        raise ValueError(f"Unknown model_size: {model_size}")

    active_dims = dims[:backbone_stages]
    active_num_blocks = num_blocks[:backbone_stages]
    
    backbone_out_keys = [f'C{i}' for i in range(backbone_stages)]
    fpn_in_keys = backbone_out_keys[-fpn_stages:]
    fpn_in_channels_map = {key: active_dims[backbone_out_keys.index(key)] for key in fpn_in_keys}
    
    cfg = {
        'model': {
            'backbone': {
                'features_only': True, 'input_channels': 1, 'target_seq_len': 1024,
                'partition_ratio': 16, 'drop_path_rate': 0.1, 'dims': active_dims,
                'num_blocks': active_num_blocks, 'out_indices': list(range(backbone_stages)),
                'out_keys': backbone_out_keys,
                'dim_head': dim_head,
            },
            'neck': { 
                'depthwise': False, 'act': 'silu', 'depth': neck_depth,
                'in_keys': fpn_in_keys, 'in_channels': fpn_in_channels_map,
            },
            'head': { 
                'name': 'MultiScaleRegressionHead', 'out_features': 2, 
                'mid_features_ratio': 0.5, 'in_channels': fpn_in_channels_map,
            }
        }
    }
    
    return OmegaConf.create(cfg)


class LidarRegressor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        
        self.backbone = MaxT1d(cfg.model.backbone)
        self.neck = PAFPN1d(cfg.model.neck)
        
        head_cfg = cfg.model.head
        if head_cfg.name == 'MultiScaleRegressionHead':
            self.head = _MultiScaleHead(head_cfg)
        else:
            raise ValueError(f"Unknown head name: {head_cfg.name}")
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        all_features = self.backbone(x)
        
        neck_input = {key: all_features[key] for key in self.neck.in_keys}
        neck_features = self.neck(neck_input)
        
        predictions = self.head(neck_features)
        
        return predictions