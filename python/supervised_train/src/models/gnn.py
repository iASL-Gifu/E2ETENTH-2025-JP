import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class LidarGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pool_method='mean'):
        super(LidarGCN, self).__init__()
        
        # --- GCN Layers (次元圧縮を含む構造) ---
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        self.conv4 = GCNConv(hidden_dim // 4, hidden_dim // 8)

        # --- Poolingの選択 ---
        self.pool_method = pool_method
        
        # --- MLPによる予測 ---
        self.fc1 = torch.nn.Linear(hidden_dim // 8, hidden_dim // 16)
        self.fc2 = torch.nn.Linear(hidden_dim // 16, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # --- 4層のGCN処理 ---
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        
        # --- プーリング処理 (Mean or Max) ---
        if self.pool_method == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_method == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError("Invalid pool method")
        
        # --- MLPによる予測 ---
        x = F.relu(self.fc1(x))
        action = self.fc2(x)  # 出力: [steer, speed]
        
        return action
