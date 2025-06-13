import torch

class RnnStateManager:
    """
    StatefulなRNNの隠れ状態を管理するクラス。
    バッチ間の状態の引き継ぎを担う。
    """
    def __init__(self, device):
        self.device = device
        self._states = None # (h_n, c_n) タプルを保持

    def reset_states(self, batch_size):
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
        # h_n と c_n の両方に対して処理を行う
        h_n, c_n = self._states
        
        # is_first_seqの形状を [B] -> [1, B, 1] にブロードキャスト可能にする
        reset_mask = is_first_seq.view(1, -1, 1).to(self.device)

        # マスクを適用してリセット
        # is_first_seqがTrueの箇所は0になり、Falseの箇所は元の値が維持される
        h_n = h_n * (~reset_mask)
        c_n = c_n * (~reset_mask)

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