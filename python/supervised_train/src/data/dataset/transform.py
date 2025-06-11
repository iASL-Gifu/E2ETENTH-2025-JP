import numpy as np

class SeqToSeqTransform:
    """
    LidarSeqToSeqDataset用のデータ前処理クラス。
    LiDARスキャンの正規化・ダウンサンプリングと、アクションのクリッピングを行う。
    """
    def __init__(self, range_max=30.0, base_num=1081, downsample_num=181):
        """
        :param range_max: Lidarの最大距離 (m)
        :param base_num: 元のLidarスキャンのサンプル数
        :param downsample_num: ダウンサンプリング後のサンプル数
        """
        self.range_max = range_max
        # ダウンサンプリング用のインデックスを事前に計算
        self.sample_indices = np.round(np.linspace(0, base_num - 1, downsample_num)).astype(int)

    def __call__(self, sample: dict) -> dict:
        """
        データセットから取得したサンプルの辞書を受け取り、前処理を行う。
        :param sample: {'scan_seq': ..., 'prev_action_seq': ..., 'target_action_seq': ...} の辞書
        :return: 前処理後のデータを持つ新しい辞書
        """
        # --- scan_seq の前処理 ---
        scan_seq = sample['scan_seq']
        # 1. 距離をクリッピングして正規化
        scan_seq = np.clip(scan_seq, 0, self.range_max) / self.range_max
        # 2. ダウンサンプリング
        processed_scan_seq = scan_seq[:, self.sample_indices]

        # --- prev_action_seq の前処理 ---
        prev_action_seq = sample['prev_action_seq'].copy()
        np.clip(prev_action_seq[:, 0], -1.0, 1.0, out=prev_action_seq[:, 0]) # steer ([:, 0]) を [-1, 1] にクリッピング
        np.clip(prev_action_seq[:, 1], -1.0, 1.0, out=prev_action_seq[:, 1]) # speed ([:, 1]) を [-1, 1] にクリッピング (必要に応じて範囲を調整)
        
        # --- target_action_seq の前処理 ---
        target_action_seq = sample['target_action_seq'].copy()
        np.clip(target_action_seq[:, 0], -1.0, 1.0, out=target_action_seq[:, 0]) # steer ([:, 0]) を [-1, 1] にクリッピング
        np.clip(target_action_seq[:, 1], -1.0, 1.0, out=target_action_seq[:, 1]) # speed ([:, 1]) を [-1, 1] にクリッピング
        
        return {
            'scan_seq': processed_scan_seq,
            'prev_action_seq': prev_action_seq,
            'target_action_seq': target_action_seq
        }