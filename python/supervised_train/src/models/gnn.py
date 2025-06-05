import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

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


class LidarGAT(torch.nn.Module):
    """
    GAT (Graph Attention Network) を使用したLiDAR点群からの制御量予測モデル。

    Args:
        input_dim (int): 入力特徴量の次元数。
        hidden_dim (int): 隠れ層の基本次元数。
        output_dim (int): 出力次元数 (例: Steer, Speedで2)。
        heads (int): GAT層で使用するアテンションヘッドの数。
        dropout_rate (float): ドロップアウト率。
        pool_method (str): プーリング手法 ('mean' or 'max')。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout_rate=0.5, pool_method='mean'):
        super(LidarGAT, self).__init__()
        self.dropout_rate = dropout_rate
        
        # --- GAT Layers (アテンション機構を持つ構造) ---
        # GATでは、複数のアテンションヘッドの出力を連結(concat)するため、
        # 次の層の入力次元は前の層の出力次元 * ヘッド数になる。
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
        
        # 中間層
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim // 2, heads=heads, dropout=dropout_rate)
        self.conv3 = GATConv((hidden_dim // 2) * heads, hidden_dim // 4, heads=heads, dropout=dropout_rate)
        
        # 最終GAT層
        # 次のMLP層に渡すため、ヘッドからの出力を平均化して次元をまとめる (concat=False)
        # あるいはヘッド数を1にする方法もある。今回は後者を採用し、次元をシンプルに保つ。
        self.conv4 = GATConv((hidden_dim // 4) * heads, hidden_dim // 8, heads=1, concat=True, dropout=dropout_rate)

        # --- Poolingの選択 ---
        self.pool_method = pool_method
        
        # --- MLPによる予測 ---
        self.fc1 = torch.nn.Linear(hidden_dim // 8, hidden_dim // 16)
        self.fc2 = torch.nn.Linear(hidden_dim // 16, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # --- 4層のGAT処理 ---
        # GATの論文では活性化関数にELUが使われることが多い
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv3(x, edge_index))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv4(x, edge_index)) # 最終GAT層
        
        # --- プーリング処理 (Mean or Max) ---
        if self.pool_method == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_method == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError("Invalid pool method")
        
        # --- MLPによる予測 ---
        x = F.relu(self.fc1(x)) # MLP部分ではReLUも一般的
        action = self.fc2(x)  # 出力: [steer, speed]
        
        return action