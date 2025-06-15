import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from omegaconf import DictConfig

from .blocks import BaseConv1d, DWConv1d, CSPLayer1d

class PAFPN1d(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        in_channels = cfg.in_channels
        self.in_keys = sorted(in_channels.keys())
        num_stages = len(in_channels)
        
        Conv = DWConv1d if cfg.depthwise else BaseConv1d
        num_csp_blocks = max(round(3 * cfg.depth), 1)

        # Top-down path (FPN)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for i in range(num_stages - 1, 0, -1):
            key_deep, key_shallow = self.in_keys[i], self.in_keys[i-1]
            self.lateral_convs.append(BaseConv1d(in_channels[key_deep], in_channels[key_shallow], 1, 1, act=cfg.act))
            self.fpn_blocks.append(CSPLayer1d(2 * in_channels[key_shallow], in_channels[key_shallow], n=num_csp_blocks, shortcut=False, depthwise=cfg.depthwise, act=cfg.act))
            
        # Bottom-up path (PAN)
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for i in range(num_stages - 1):
            key_shallow, key_deep = self.in_keys[i], self.in_keys[i+1]
            self.downsample_convs.append(Conv(in_channels[key_shallow], in_channels[key_shallow], 3, 2, act=cfg.act))
            
            pan_in_channels = in_channels[key_shallow] + in_channels[key_deep]
            self.pan_blocks.append(CSPLayer1d(pan_in_channels, in_channels[key_deep], n=num_csp_blocks, shortcut=False, depthwise=cfg.depthwise, act=cfg.act))

        self.out_channels_map = {key: in_channels[key] for key in self.in_keys}

    @torch.jit.export
    def upsample_feature(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode='nearest')

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features_permuted = {k: v.permute(0, 2, 1) for k, v in features.items()}
        
        # 1. Top-down path
        inner_features = [features_permuted[self.in_keys[-1]]]
        # lateral_convs と fpn_blocks を同時に enumerate するように変更
        for i, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            idx = len(self.in_keys) - 2 - i
            lat_feat = lateral_conv(inner_features[-1]) 
            f_in = features_permuted[self.in_keys[idx]]
            f_out = torch.cat([self.upsample_feature(lat_feat), f_in], 1)
            inner_features.append(fpn_block(f_out)) # ここを修正: fpn_blocks[i] ではなく fpn_block を直接呼び出す
        inner_features.reverse()
        
        # 2. Bottom-up path
        # こちらのループも同様に修正が必要になる可能性があります
        pan_outputs = [inner_features[0]]
        # downsample_convs と pan_blocks を同時に enumerate するように変更
        for i, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            p_in = downsample_conv(pan_outputs[-1]) # ここを修正: downsample_convs[i] ではなく downsample_conv を直接呼び出す
            p_out = torch.cat([p_in, inner_features[i+1]], 1)
            pan_outputs.append(pan_block(p_out)) # ここを修正: pan_blocks[i] ではなく pan_block を直接呼び出す
            
        return {key: val for key, val in zip(self.in_keys, pan_outputs)}
