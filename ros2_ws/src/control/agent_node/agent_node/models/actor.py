import torch
import torch.nn as nn

from .backbone import TinyLidarBackbone

def get_actor(state_dim, action_dim, hidden_dim, policy_type='mlp', log_std_min=-20, log_std_max=2):
    """
    アクターモデルを取得する関数
    :param state_dim: 状態空間の次元数
    :param action_dim: 行動空間の次元数
    :param hidden_dim: 隠れ層の次元数
    :param policy_type: 'mlp' または '1dconv'
    :param log_std_min: 対数標準偏差の最小値
    :param log_std_max: 対数標準偏差の最大値
    :return: アクターモデル
    """
    if policy_type == 'mlp':
        return MLPPolicy(state_dim, action_dim, hidden_dim, log_std_min, log_std_max)
    elif policy_type == '1dconv':
        return ConvPolicy(state_dim, action_dim, hidden_dim, log_std_min, log_std_max)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

class MLPPolicy(nn.Module):
    """
    ガウス政策ネットワーク (連続行動空間)
    出力は平均と対数分散を返し、sample()でtanh補正後の行動とログ確率を返す。
    """
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)
    
class ConvPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.lidar_backbone = TinyLidarBackbone(input_dim=state_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(self.lidar_backbone.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.lidar_backbone(state)
        x = self.net(features)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)
