import torch
import torch.nn as nn
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
        if isinstance(data, list):
            data = data[-1]
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


class LidarGcnLstmNet(nn.Module):
    """
    GCNとLSTMを組み合わせた、自己完結した時系列モデル。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 lstm_hidden_dim=128, lstm_layers=1, pool_method='mean'):
        super(LidarGcnLstmNet, self).__init__()
        
        # --- GCN層の定義 ---
        self.gcn_conv1 = GCNConv(input_dim, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.gcn_conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        self.gcn_conv4 = GCNConv(hidden_dim // 4, hidden_dim // 8)
        self.pool_method = pool_method
        
        # GCNエンコーダ部分の出力次元
        gcn_embedding_dim = hidden_dim // 8
        
        # --- LSTM層の定義 ---
        self.lstm = nn.LSTM(
            input_size=gcn_embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # --- 全結合層（MLP）の定義 ---
        self.fc_out = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, data_sequence):
        """
        Args:
            data_sequence (list of torch_geometric.data.Batch): 
                各要素が1タイムステップ分のグラフバッチであるリスト。
        """
        if not isinstance(data_sequence, list):
            data_sequence = [data_sequence]

        # --- Step 1: 各タイムステップのグラフからGCNで特徴ベクトルを抽出 ---
        gcn_embeddings = []
        for graph_batch in data_sequence:
            x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
            
            # GCN処理
            x = F.relu(self.gcn_conv1(x, edge_index))
            x = F.relu(self.gcn_conv2(x, edge_index))
            x = F.relu(self.gcn_conv3(x, edge_index))
            x = F.relu(self.gcn_conv4(x, edge_index))
            
            # プーリング処理
            if self.pool_method == 'mean':
                embedding = global_mean_pool(x, batch)
            elif self.pool_method == 'max':
                embedding = global_max_pool(x, batch)
            else:
                raise ValueError("Invalid pool method")
            
            gcn_embeddings.append(embedding)
            
        # --- Step 2: 特徴ベクトルを時間軸に沿って1つのテンソルに束ねる ---
        # 形状: (batch_size, sequence_length, gcn_embedding_dim)
        gnn_sequence = torch.stack(gcn_embeddings, dim=1)

        # --- Step 3: LSTMに特徴シーケンスを入力 ---
        # lstm_outの形状: (batch_size, sequence_length, lstm_hidden_dim)
        lstm_out, _ = self.lstm(gnn_sequence)
        
        # --- Step 4: シーケンスの最後の出力を使って予測 ---
        # last_lstm_outの形状: (batch_size, lstm_hidden_dim)
        last_lstm_out = lstm_out[:, -1, :]
        
        # --- Step 5: 最終的な制御量を出力 ---
        action = self.fc_out(last_lstm_out)
        
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
        if isinstance(data, list):
            data = data[-1]
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
    
class LidarGatLstmNet(nn.Module):
    """
    GATとLSTMを組み合わせた、自己完結した時系列モデル。
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 lstm_hidden_dim=128, lstm_layers=1, heads=8, 
                 dropout_rate=0.5, pool_method='mean'):
        super(LidarGatLstmNet, self).__init__()
        
        # --- GAT層の定義 ---
        self.dropout_rate = dropout_rate
        self.gat_conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
        self.gat_conv2 = GATConv(hidden_dim * heads, hidden_dim // 2, heads=heads, dropout=dropout_rate)
        self.gat_conv3 = GATConv((hidden_dim // 2) * heads, hidden_dim // 4, heads=heads, dropout=dropout_rate)
        self.gat_conv4 = GATConv((hidden_dim // 4) * heads, hidden_dim // 8, heads=1, concat=True, dropout=dropout_rate)
        self.pool_method = pool_method
        
        # GATエンコーダ部分の出力次元
        gat_embedding_dim = hidden_dim // 8
        
        # --- LSTM層の定義 ---
        self.lstm = nn.LSTM(
            input_size=gat_embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # --- 全結合層（MLP）の定義 ---
        self.fc_out = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, data_sequence):
        """
        Args:
            data_sequence (list of torch_geometric.data.Batch): 
                各要素が1タイムステップ分のグラフバッチであるリスト。
        """
        if not isinstance(data_sequence, list):
            data_sequence = [data_sequence]

        # --- Step 1: 各タイムステップのグラフからGATで特徴ベクトルを抽出 ---
        gat_embeddings = []
        for graph_batch in data_sequence:
            x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

            # GAT処理
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.elu(self.gat_conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.elu(self.gat_conv2(x, edge_index))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.elu(self.gat_conv3(x, edge_index))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.elu(self.gat_conv4(x, edge_index))
            
            # プーリング処理
            if self.pool_method == 'mean':
                embedding = global_mean_pool(x, batch)
            elif self.pool_method == 'max':
                embedding = global_max_pool(x, batch)
            else:
                raise ValueError("Invalid pool method")

            gat_embeddings.append(embedding)
            
        # --- Step 2: 特徴ベクトルを時間軸に沿って1つのテンソルに束ねる ---
        gnn_sequence = torch.stack(gat_embeddings, dim=1)

        # --- Step 3: LSTMに特徴シーケンスを入力 ---
        lstm_out, _ = self.lstm(gnn_sequence)
        
        # --- Step 4: シーケンスの最後の出力を使って予測 ---
        last_lstm_out = lstm_out[:, -1, :]
        
        # --- Step 5: 最終的な制御量を出力 ---
        action = self.fc_out(last_lstm_out)
        
        return action