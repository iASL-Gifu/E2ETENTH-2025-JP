import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    学習時(train)と評価時(eval)で出力形式が変わるモデル。
    - train時: (Batch, SequenceLength, output_dim) を出力
    - eval時 : (Batch, output_dim) を出力
    """
    def __init__(self, input_dim, output_dim, lstm_hidden_dim=128, lstm_layers=1):
        # ... __init__ の中身は変更なし ...
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

    def forward(self, x):
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

        # --- LSTM (共通部分) ---
        lstm_out, (h_n, c_n) = self.lstm(x)

        # self.training は model.train() / .eval() で切り替わる
        if self.training:
            # --- 学習モードの場合 ---
            # FC層に全シーケンスを渡し、(B, T, F) の出力を得る
            x = lstm_out.contiguous().view(batch_size * seq_len, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            # 出力の形状を (Batch, SequenceLength, output_dim) に戻す
            x = x.view(batch_size, seq_len, -1)
        else:
            # --- 評価・推論モードの場合 ---
            # シーケンスの最後の出力だけを取り出してFC層に渡す
            last_lstm_out = lstm_out[:, -1, :] # Shape: (Batch, lstm_hidden_dim)
            x = F.relu(self.fc1(last_lstm_out))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            # 出力は (Batch, output_dim) の2次元テンソルになる
        
        return x