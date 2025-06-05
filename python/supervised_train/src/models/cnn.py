import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLidarNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)   # -> (24, L1)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)   # -> (36, L2)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)   # -> (48, L3)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)             # -> (64, L4)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)             # -> (64, L5)

        # フラット化後の出力サイズを動的に推定
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)  # (B, C, L)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

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


class TinyLidarLstmNet(nn.Module):
    """
    TinyLidarNetにLSTMレイヤーを追加した時系列対応モデル。
    """
    def __init__(self, input_dim, output_dim, lstm_hidden_dim=128, lstm_layers=1):
        super().__init__()

        # --- 1. CNN Feature Extractor (特徴抽出器) ---
        # この部分はTinyLidarNetと同じ
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)

        # CNN部分の出力サイズを動的に計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            self.flatten_dim = cnn_out.view(1, -1).shape[1]

        # --- 2. LSTM (時間的特徴の学習) ---
        # CNNの出力を受けるLSTMレイヤーを追加
        self.lstm = nn.LSTM(
            input_size=self.flatten_dim,  # CNNからの特徴ベクトルの次元
            hidden_size=lstm_hidden_dim,  # LSTMの隠れ状態の次元
            num_layers=lstm_layers,
            batch_first=True  # 入力データの形式を (batch, seq, feature) にする
        )

        # --- 3. Fully Connected Layers (最終的な出力) ---
        # LSTMの出力を受けるように入力次元を変更
        self.fc1 = nn.Linear(lstm_hidden_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    def forward(self, x):
        # 入力xの想定形状: (Batch, SequenceLength, Length)
        # 例: (64, 10, 181)
        batch_size, seq_len, length = x.shape

        # CNNは (B, C, L) の入力を期待するため、バッチとシーケンスの次元をまとめる
        # (B, T, L) -> (B * T, L)
        x = x.view(batch_size * seq_len, length)
        # (B * T, L) -> (B * T, 1, L)
        x = x.unsqueeze(1)

        # --- CNNによる特徴抽出 (各フレームごと) ---
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # (B * T, Channels, L_out) -> (B * T, flatten_dim)
        x = x.view(batch_size * seq_len, -1)

        # --- LSTMへの入力のために形状をシーケンスに戻す ---
        # (B * T, flatten_dim) -> (B, T, flatten_dim)
        x = x.view(batch_size, seq_len, self.flatten_dim)

        # --- LSTMによる時間的特徴の学習 ---
        # self.lstmは (出力のシーケンス, (最後の隠れ状態, 最後のセル状態)) を返す
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 最後のタイムステップの隠れ状態 h_n を使って予測を行う
        # h_n の形状は (num_layers, batch_size, hidden_dim) なので、最後のレイヤーを取り出す
        x = h_n[-1, :, :]
        
        # --- FC層で最終的な出力を計算 ---
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x)) # 出力を [-1, 1] に制限
        return x