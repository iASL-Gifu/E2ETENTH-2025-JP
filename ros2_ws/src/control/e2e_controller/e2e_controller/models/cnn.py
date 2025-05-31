import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLidarNet(nn.Module):
    def __init__(self, input_length):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)   # -> (24, L1)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)   # -> (36, L2)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)   # -> (48, L3)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)             # -> (64, L4)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)             # -> (64, L5)

        # フラット化後の出力サイズを動的に推定
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)  # (B, C, L)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L) に変換
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # 出力を [-1, 1] に制限
        return x