import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.lstm import ConvLSTM1dLayer

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