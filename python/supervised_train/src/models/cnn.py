import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .layers.lstm import ConvLSTM1dLayer
from .layers.pos_encode import PositionalEncoding

### 1. 汎用的な重み初期化関数 ###
def _init_weights(m):
    """
    モジュールに応じた重み初期化を適用する関数
    """
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        # 活性化関数がReLUなので、He初期化 (Kaiming Normal) を使用
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            # バイアスは0で初期化
            init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # 入力-隠れ状態間の重み (Xavier Uniform)
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # 隠れ-隠れ状態間の重み (Orthogonal: 直交行列は再帰処理で勾配消失/爆発を防ぐのに効果的)
                init.orthogonal_(param.data)
            elif 'bias' in name:
                # バイアスは0で初期化
                param.data.fill_(0)
                # (Tips) forget gateのバイアスを1にすると、長期的な依存関係を学習しやすくなることがある
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)

### 2. 各モデルクラスに初期化処理を適用 ###

class TinyLidarNet(nn.Module):
    def __init__(self, input_dim, output_dim):
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
        
        # 重み初期化を適用
        self.apply(_init_weights)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :] 
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


class TinyLidarLstmNet(nn.Module):
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

        # 重み初期化を適用
        self.apply(_init_weights)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, length = x.shape
        x = x.view(batch_size * seq_len, length)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(batch_size, seq_len, self.flatten_dim)
        lstm_out, hidden = self.lstm(x, hidden)
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
        return x, hidden
    

class TinyLidarConvLstmNet(nn.Module):
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

        # 重み初期化を適用
        self.apply(_init_weights)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, length = x.shape
        x = x.view(batch_size * seq_len, length)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(batch_size, seq_len, self.cnn_channels, self.cnn_length)
        lstm_out, last_hidden = self.conv_lstm(x, hidden)
        h_n, c_n = last_hidden
        if self.training:
            x = lstm_out.contiguous().view(batch_size * seq_len, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            x = x.view(batch_size, seq_len, -1)
        else:
            last_out = h_n.view(batch_size, -1)
            x = F.relu(self.fc1(last_out))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
        return x, last_hidden
    

class TinyLidarActionNet(nn.Module):
    def __init__(self, input_dim, output_dim, action_dim=2):
        super().__init__()
        self.action_dim = action_dim
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            self.flatten_dim = cnn_out.view(1, -1).shape[1]
        self.fc1 = nn.Linear(self.flatten_dim + self.action_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        # 重み初期化を適用
        self.apply(_init_weights)

    def forward(self, x, pre_action):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        cnn_features = F.relu(self.conv1(x))
        cnn_features = F.relu(self.conv2(cnn_features))
        cnn_features = F.relu(self.conv3(cnn_features))
        cnn_features = F.relu(self.conv4(cnn_features))
        cnn_features = F.relu(self.conv5(cnn_features))
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        if pre_action.dim() == 3:
            pre_action = pre_action.squeeze(1)
        combined_features = torch.cat((cnn_features, pre_action), dim=1)
        out = F.relu(self.fc1(combined_features))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = torch.tanh(self.fc4(out))
        return out


class TinyLidarActionLstmNet(nn.Module):
    def __init__(self, input_dim, output_dim, action_dim=2, lstm_hidden_dim=128, lstm_layers=1):
        super().__init__()
        self.action_dim = action_dim
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
            input_size=self.flatten_dim + self.action_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(lstm_hidden_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        # 重み初期化を適用
        self.apply(_init_weights)

    def forward(self, x, pre_action, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, length = x.shape
        cnn_features = x.view(batch_size * seq_len, length)
        cnn_features = cnn_features.unsqueeze(1)
        cnn_features = F.relu(self.conv1(cnn_features))
        cnn_features = F.relu(self.conv2(cnn_features))
        cnn_features = F.relu(self.conv3(cnn_features))
        cnn_features = F.relu(self.conv4(cnn_features))
        cnn_features = F.relu(self.conv5(cnn_features))
        cnn_features = cnn_features.view(batch_size, seq_len, self.flatten_dim)
        if pre_action.dim() == len(cnn_features.shape) -1:
             pre_action = pre_action.unsqueeze(-1)
        lstm_input = torch.cat((cnn_features, pre_action), dim=2)
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        if self.training:
            out = lstm_out.contiguous().view(batch_size * seq_len, -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
            out = out.view(batch_size, seq_len, -1)
        else:
            last_lstm_out = lstm_out[:, -1, :]
            out = F.relu(self.fc1(last_lstm_out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
        return out, hidden
    

class TinyLidarActionConvLstmNet(nn.Module):
    def __init__(self, input_dim, output_dim, action_dim=2, dws_conv_kernel_size=5):
        super().__init__()
        self.action_dim = action_dim
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

        # 重み初期化を適用
        self.apply(_init_weights)

    def forward(self, x, pre_action, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, length = x.shape
        cnn_features = x.view(batch_size * seq_len, length)
        cnn_features = cnn_features.unsqueeze(1)
        cnn_features = F.relu(self.conv1(cnn_features))
        cnn_features = F.relu(self.conv2(cnn_features))
        cnn_features = F.relu(self.conv3(cnn_features))
        cnn_features = F.relu(self.conv4(cnn_features))
        cnn_features = F.relu(self.conv5(cnn_features))
        cnn_features = cnn_features.view(batch_size, seq_len, self.cnn_channels, self.cnn_length)
        action_map = pre_action.unsqueeze(-1).expand(-1, -1, -1, self.cnn_length)
        conv_lstm_input = torch.cat((cnn_features, action_map), dim=2)
        lstm_out, last_hidden = self.conv_lstm(conv_lstm_input, hidden)
        h_n, c_n = last_hidden
        if self.training:
            out = lstm_out.contiguous().view(batch_size * seq_len, -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
            out = out.view(batch_size, seq_len, -1)
        else:
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
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            flatten_dim = cnn_out.view(1, -1).shape[1]
        self.input_projection = nn.Linear(flatten_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(d_model, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

        # 重み初期化を適用
        self.apply(_init_weights)

    def forward(self, x):
        batch_size, seq_len, length = x.shape
        cnn_features = x.view(batch_size * seq_len, length)
        cnn_features = F.relu(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(cnn_features.unsqueeze(1)))))))
        cnn_flattened = cnn_features.view(batch_size * seq_len, -1)
        projected_features = self.input_projection(cnn_flattened)
        transformer_input = projected_features.view(batch_size, seq_len, -1)
        transformer_input = self.pos_encoder(transformer_input)
        transformer_out = self.transformer_encoder(transformer_input)
        if self.training:
            out = transformer_out.contiguous().view(batch_size * seq_len, -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
            out = out.view(batch_size, seq_len, -1)
        else:
            last_out = transformer_out[:, -1, :]
            out = F.relu(self.fc1(last_out))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = torch.tanh(self.fc4(out))
        return out