from typing import Optional, Tuple
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class DWSConvLSTM1d(nn.Module):
    """LSTM with (depthwise-separable) Conv option in NCHW [channel-first] format, but for 1D data.
    Input/Output format is (N, C, L).
    """

    def __init__(self,
                 dim: int,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        
        # Conv2d -> Conv1d に変更
        self.conv_dws = nn.Conv1d(in_channels=conv3x3_dws_dim,
                                  out_channels=conv3x3_dws_dim,
                                  kernel_size=dws_conv_kernel_size,
                                  padding=dws_conv_kernel_size // 2,
                                  groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        
        # Conv2d -> Conv1d に変更
        self.conv1x1 = nn.Conv1d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x: th.Tensor, h_and_c_previous: Optional[Tuple[th.Tensor, th.Tensor]] = None) \
            -> Tuple[th.Tensor, th.Tensor]:
        """
        Processes a single time step.
        :param x: (N, C, L) - Input for the current time step.
        :param h_and_c_previous: ((N, C, L), (N, C, L)) - Hidden and cell states from the previous time step.
        :return: ((N, C, L), (N, C, L)) - New hidden and cell states.
        """
        if h_and_c_previous is None:
            # generate zero states
            hidden = th.zeros_like(x)
            cell = th.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv_dws(h_tm1)
        
        xh = th.cat((x, h_tm1), dim=1) # channel-dimで連結
        
        if not self.conv_only_hidden:
            xh = self.conv_dws(xh)
            
        mix = self.conv1x1(xh)

        # ゲートとセル入力を分離
        gates, cell_input = th.tensor_split(mix, [self.dim * 3], dim=1)
        
        gates = th.sigmoid(gates)
        forget_gate, input_gate, output_gate = th.tensor_split(gates, 3, dim=1)
        
        cell_input = self.cell_update_dropout(th.tanh(cell_input))

        # セル状態と隠れ状態を更新
        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * th.tanh(c_t)

        return h_t, c_t
    
class ConvLSTM1dLayer(nn.Module):
    """A wrapper to process a sequence of 1D data using the ConvLSTM1d cell."""
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.cell = DWSConvLSTM1d(dim=dim, **kwargs)

    def forward(self, x: th.Tensor, h_and_c_initial: Optional[Tuple[th.Tensor, th.Tensor]] = None):
        """
        :param x: (Batch, SeqLen, Channels, Length)
        :param h_and_c_initial: Optional initial hidden and cell states
        :return: A tuple of (all_hidden_states, (last_hidden_state, last_cell_state))
        """
        batch_size, seq_len, _, _ = x.shape
        
        # 初期の隠れ状態とセル状態のタプルをそのまま使う
        h_c = h_and_c_initial

        # 各タイムステップの隠れ状態を保存するリスト
        outputs = []
        
        # シーケンスの長さにわたってループ
        for t in range(seq_len):
            # x_t の形状: (Batch, Channels, Length)
            x_t = x[:, t, :, :]
            # h_c が None であれば、そのまま None が渡される
            h_c = self.cell(x_t, h_c)
            # h_c は (h, c) のタプルなので、h を取り出して保存
            outputs.append(h_c[0])

        # (SeqLen, Batch, Channels, Length) -> (Batch, SeqLen, Channels, Length)
        all_hidden_states = torch.stack(outputs, dim=1)
        
        return all_hidden_states, h_c
