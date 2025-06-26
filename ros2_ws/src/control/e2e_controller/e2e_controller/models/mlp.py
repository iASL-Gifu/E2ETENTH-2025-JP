import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        シンプルな多層パーセプトロン（MLP）モデル

        Args:
            input_dim (int): 入力次元数 (LiDARの点数など。例: 1081)
            output_dim (int): 出力次元数 (例: 2)
        """
        super().__init__()

        # --- 各層の定義 ---
        # 隠れ層のユニット数を定義します。この値は調整可能です。
        hidden1_dim = 256
        hidden2_dim = 128

        # 全結合層 (Linear Layer)
        # 畳み込み層の代わりに、全結合層のみで構成します。
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        """
        順伝播処理

        Args:
            x (torch.Tensor): 入力データ。形状は (バッチサイズ, input_dim) を想定。

        Returns:
            torch.Tensor: モデルの出力
        """
        # もし入力が3次元 (B, T, L) なら、最後のタイムステップのデータだけを取り出す
        if x.dim() == 3:
            # (B, T, L) -> (B, L)
            x = x[:, -1, :] 
        # MLPでは通常、(バッチサイズ, 特徴量数) の2次元テンソルを入力とするため、
        # 元のコードにあった3次元入力の処理は不要になります。

        # 活性化関数 ReLU を通して次の層へ
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 出力層の活性化関数は、元のモデルに合わせて tanh を使用
        # tanh は出力を -1 から 1 の範囲に収めます。
        x = torch.tanh(self.fc3(x))

        return x