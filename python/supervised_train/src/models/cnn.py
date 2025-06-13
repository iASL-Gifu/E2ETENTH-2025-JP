import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.lstm import ConvLSTM1dLayer
from .layers.pos_encode import PositionalEncoding

class TinyLidarNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        # ... __init__ の中身は変更なし ...
        super().__init__()

        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    def forward(self, x):
        # 入力xの想定形状: (Batch, SequenceLength, Length) または (Batch, Length)
        
        # もし入力が3次元 (B, T, L) なら、最後のタイムステップのデータだけを取り出す
        if x.dim() == 3:
            # (B, T, L) -> (B, L)
            x = x[:, -1, :] 
        
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
        x = torch.tanh(self.fc4(x))
        return x


class TinyLidarLstmNet(nn.Module):
    # ... __init__ は変更なし ...
    def __init__(self, input_dim, output_dim, lstm_hidden_dim=128, lstm_layers=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            self.flatten_dim = cnn_out.view(1, -1).shape[1]
        self.lstm = nn.LSTM(
            input_size=self.flatten_dim, hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers, batch_first=True
        )
        self.fc1 = nn.Linear(lstm_hidden_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    # hiddenを引数で受け取り、戻り値で返すように変更
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, length = x.shape
        
        # --- CNN Feature Extraction (共通部分) ---
        x = x.view(batch_size * seq_len, length)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(batch_size * seq_len, -1)
        x = x.view(batch_size, seq_len, self.flatten_dim)

        # --- LSTM (hiddenを受け取るように変更) ---
        lstm_out, hidden = self.lstm(x, hidden) # hiddenは(h_n, c_n)のタプル

        if self.training:
            x = lstm_out.contiguous().view(batch_size * seq_len, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            x = x.view(batch_size, seq_len, -1)
        else:
            last_lstm_out = lstm_out[:, -1, :]
            x = F.relu(self.fc1(last_lstm_out))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
        
        # 計算後のhidden stateを返す
        return x, hidden
    
class TinyLidarConvLstmNet(nn.Module):
    """
    標準のLSTMの代わりにConvLSTM1dLayerを使用するモデル。
    - train時: (Batch, SequenceLength, output_dim) を出力
    - eval時 : (Batch, output_dim) を出力
    - 状態の受け渡しに対応
    """
    def __init__(self, input_dim, output_dim, dws_conv_kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            self.cnn_channels = cnn_out.shape[1]
            self.cnn_length = cnn_out.shape[2]
            self.flatten_dim = self.cnn_channels * self.cnn_length

        self.conv_lstm = ConvLSTM1dLayer(
            dim=self.cnn_channels, 
            dws_conv_kernel_size=dws_conv_kernel_size
        )
        
        self.fc1 = nn.Linear(self.flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    ### ここからが修正箇所 ###
    def forward(self, x, hidden=None): # hiddenを引数で受け取る
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, length = x.shape
        
        # --- CNN Feature Extraction ---
        x = x.view(batch_size * seq_len, length)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(batch_size, seq_len, self.cnn_channels, self.cnn_length)

        # lstm_out: (B, T, C, L_cnn),  last_hidden: ((B, C, L_cnn), (B, C, L_cnn))
        lstm_out, last_hidden = self.conv_lstm(x, hidden)
        h_n, c_n = last_hidden # 最後の隠れ状態とセル状態

        if self.training:
            # --- 学習モードの場合 ---
            x = lstm_out.contiguous().view(batch_size * seq_len, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            x = x.view(batch_size, seq_len, -1)
        else:
            # --- 評価・推論モードの場合 ---
            last_out = h_n.view(batch_size, -1)
            x = F.relu(self.fc1(last_out))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
        
        # 推論結果と最後の状態をタプルで返す
        return x, last_hidden
    

class TinyLidarActionLstmNet(nn.Module):
    """
    LiDARスキャンと前のアクションを入力とするモデル
    """
    def __init__(self, input_dim, output_dim, action_dim=2, lstm_hidden_dim=128, lstm_layers=1):
        super().__init__()
        # --- (変更点) actionの次元を保持 ---
        self.action_dim = action_dim

        # --- CNN層 (変更なし) ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            self.flatten_dim = cnn_out.view(1, -1).shape[1]

        # --- LSTMのinput_sizeを CNN特徴量 + action次元 に変更 ---
        self.lstm = nn.LSTM(
            input_size=self.flatten_dim + self.action_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(lstm_hidden_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    # forwardの引数に pre_action を追加 ---
    def forward(self, x, pre_action, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, length = x.shape

        # --- CNN Feature Extraction (LiDARデータのみを処理) ---
        cnn_features = x.view(batch_size * seq_len, length)
        cnn_features = cnn_features.unsqueeze(1)
        cnn_features = F.relu(self.conv1(cnn_features))
        cnn_features = F.relu(self.conv2(cnn_features))
        cnn_features = F.relu(self.conv3(cnn_features))
        cnn_features = F.relu(self.conv4(cnn_features))
        cnn_features = F.relu(self.conv5(cnn_features))
        cnn_features = cnn_features.view(batch_size, seq_len, self.flatten_dim)

        
        # pre_actionの形状を (batch_size, seq_len, action_dim) に合わせる
        # 例: pre_actionが(batch_size, seq_len)ならunsqueezeで次元追加
        if pre_action.dim() == len(cnn_features.shape) -1:
             pre_action = pre_action.unsqueeze(-1)
        
        # 特徴量次元(dim=2)で連結
        lstm_input = torch.cat((cnn_features, pre_action), dim=2)

        # --- LSTM (連結した特徴量を入力) ---
        lstm_out, hidden = self.lstm(lstm_input, hidden)

        # --- 全結合層 (以降は変更なし) ---
        if self.training:
            # 訓練時: シーケンス全体の出力を計算
            out = lstm_out.contiguous().view(batch_size * seq_len, -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
            out = out.view(batch_size, seq_len, -1)
        else:
            # 推論時: シーケンスの最後のタイムステップの出力のみを計算
            last_lstm_out = lstm_out[:, -1, :]
            out = F.relu(self.fc1(last_lstm_out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))

        return out, hidden
    

class TinyLidarActionConvLstmNet(nn.Module):
    """
    LiDARスキャンと前のアクションを入力とし、ConvLSTMを使用するモデル。
    """
    def __init__(self, input_dim, output_dim, action_dim=2, dws_conv_kernel_size=5):
        super().__init__()
        # --- actionの次元を保持 ---
        self.action_dim = action_dim

        # --- CNN層  ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            self.cnn_channels = cnn_out.shape[1]
            self.cnn_length = cnn_out.shape[2]

        # --- ConvLSTMの入力チャネル数と、FC層の入力次元を更新 ---
        self.conv_lstm_input_channels = self.cnn_channels + self.action_dim
        self.conv_lstm = ConvLSTM1dLayer(
            dim=self.conv_lstm_input_channels, 
            dws_conv_kernel_size=dws_conv_kernel_size
        )
        
        self.flatten_dim = self.conv_lstm_input_channels * self.cnn_length
        self.fc1 = nn.Linear(self.flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    def forward(self, x, pre_action, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, length = x.shape
        
        # --- CNN Feature Extraction (LiDARデータのみを処理) ---
        cnn_features = x.view(batch_size * seq_len, length)
        cnn_features = cnn_features.unsqueeze(1)
        cnn_features = F.relu(self.conv1(cnn_features))
        cnn_features = F.relu(self.conv2(cnn_features))
        cnn_features = F.relu(self.conv3(cnn_features))
        cnn_features = F.relu(self.conv4(cnn_features))
        cnn_features = F.relu(self.conv5(cnn_features))
        cnn_features = cnn_features.view(batch_size, seq_len, self.cnn_channels, self.cnn_length)

        # --- pre_actionを特徴マップ形状に拡張し、連結 ---
        # pre_actionの形状を (B, T, action_dim) -> (B, T, action_dim, L_cnn) に拡張
        action_map = pre_action.unsqueeze(-1).expand(-1, -1, -1, self.cnn_length)
        
        # チャネル次元(dim=2)で連結
        conv_lstm_input = torch.cat((cnn_features, action_map), dim=2)
        
        # --- ConvLSTM (連結した特徴マップを入力) ---
        # lstm_out: (B, T, C_new, L_cnn), last_hidden: ((B, C_new, L_cnn), (B, C_new, L_cnn))
        lstm_out, last_hidden = self.conv_lstm(conv_lstm_input, hidden)
        h_n, c_n = last_hidden

        if self.training:
            # --- 学習モードの場合 ---
            out = lstm_out.contiguous().view(batch_size * seq_len, -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
            out = out.view(batch_size, seq_len, -1)
        else:
            # --- 評価・推論モードの場合 ---
            last_out = h_n.view(batch_size, -1)
            out = F.relu(self.fc1(last_out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
        
        return out, last_hidden


class TinyLidarConvTransformerNet(nn.Module):
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        # --- 1. CNN層 ---
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            flatten_dim = cnn_out.view(1, -1).shape[1]

        # CNNからの巨大な特徴量次元(flatten_dim)を、Transformerが扱いやすい次元(d_model)に圧縮する
        self.input_projection = nn.Linear(flatten_dim, d_model)
        
        # --- 2. Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # --- 3. Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # --- 4. FC層 (入力次元はd_model) ---
        self.fc1 = nn.Linear(d_model, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    def forward(self, x):
        batch_size, seq_len, length = x.shape
        
        # --- 1. CNN Feature Extraction ---
        cnn_features = x.view(batch_size * seq_len, length)
        # ... (CNNのforwardパスは変更なし) ...
        cnn_features = F.relu(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(cnn_features.unsqueeze(1)))))))
        
        # (B * T, C, L_out) -> (B * T, flatten_dim) に変形
        cnn_flattened = cnn_features.view(batch_size * seq_len, -1)
        
        # ★★★ Projection Layerを適用 ★★★
        projected_features = self.input_projection(cnn_flattened)
        
        # (B * T, d_model) -> (B, T, d_model) に変形
        transformer_input = projected_features.view(batch_size, seq_len, -1)

        # --- 2. Positional Encoding ---
        transformer_input = self.pos_encoder(transformer_input)
        
        # --- 3. Transformer Encoder ---
        transformer_out = self.transformer_encoder(transformer_input)
        
        # --- 4. FC Layers ---
        if self.training:
            out = transformer_out.contiguous().view(batch_size * seq_len, -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
            out = out.view(batch_size, seq_len, -1)
        else:
            last_out = transformer_out[:, -1, :] # (Batch, d_model)
            out = F.relu(self.fc1(last_out))
            out = F.relu(self.fc2(last_out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
        return out