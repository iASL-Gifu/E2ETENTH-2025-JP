import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLidarBackbone(nn.Module):
    def __init__(self, input_dim=1081):
        super().__init__()
        # conv 層はそのまま流用
        self.conv1 = nn.Conv1d(1, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        # ダミー入力で flatten サイズを自動計算
        dummy = torch.zeros(1, 1, input_dim)
        feat = self._forward_conv(dummy)
        self.out_dim = feat.view(1, -1).size(1)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(1)                   # → (batch, 1, input_dim)
        x = self._forward_conv(x)            # → (batch, C, L)
        return x.view(x.size(0), -1)         # → (batch, out_dim)
