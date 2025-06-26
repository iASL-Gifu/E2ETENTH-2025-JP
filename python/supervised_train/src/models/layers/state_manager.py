class RnnStateManager:
    """
    StatefulなRNNの隠れ状態を管理するクラス。
    バッチ間の状態の引き継ぎを担う。
    """
    def __init__(self, device):
        self.device = device
        self._states = None # (h_n, c_n) タプルを保持

    def reset_states(self):
        """エポック開始時に状態を完全にリセットする。"""
        self._states = None
        print("[INFO] RNN states have been reset for the new epoch.")

    def get_states_for_batch(self, is_first_seq):
        """
        現在のバッチのための状態を取得する。
        is_first_seqがTrueのサンプルに対応する状態はNoneにリセットする。
        """
        # 最初のバッチ、または状態が未設定の場合
        if self._states is None:
            return None

        # is_first_seqフラグに基づいて、特定サンプルの状態のみをリセット
        h_n, c_n = self._states
        
        # --- ▼ ここからが修正箇所 ▼ ---

        # is_first_seq (形状: [B]) を bool に変換
        reset_mask_bool = is_first_seq.to(self.device)

        # h_n と c_n の形状を調べて、適切な形状のマスクを作成
        # h_n の形状は (num_layers, B, H) または (B, C, L)
        if h_n.dim() == 3 and h_n.shape[0] == 1:
            # nn.LSTM (num_layers=1) の場合: h_n.shape は (1, B, H)
            # reset_mask の形状を (1, B, 1) にする
            reset_mask = reset_mask_bool.view(1, -1, 1)
        elif h_n.dim() == 3:
            # ConvLSTM の場合: h_n.shape は (B, C, L)
            # reset_mask の形状を (B, 1, 1) にする
            reset_mask = reset_mask_bool.view(-1, 1, 1)
        else:
            # 予期しない形状の場合のエラーハンドリング
            raise ValueError(f"Unsupported hidden state shape: {h_n.shape}")

        # マスクを適用してリセット
        # is_first_seqがTrueの箇所は 0 になり、Falseの箇所は元の値が維持される
        # ブールインデックスのNOT(~)を使うためにbool型に変換
        h_n = h_n * (~reset_mask)
        c_n = c_n * (~reset_mask)
        
        # --- ▲ ここまでが修正箇所 ▲ ---

        return (h_n, c_n)

    def save_states_from_batch(self, new_states):
        """
        モデルから出力された新しい状態を保存する。
        計算グラフから切り離すために .detach() を使用する。
        """
        if isinstance(new_states, tuple): # LSTMの場合 (h, c)
            h_n, c_n = new_states
            self._states = (h_n.detach(), c_n.detach())
        else: # GRUなどの場合
            self._states = new_states.detach()