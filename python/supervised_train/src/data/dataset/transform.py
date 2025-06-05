import numpy as np

class E2ETransform:
    def __init__(self, range_max=30.0, base_num=1081, downsample_num=181):
        """
        Lidarとsteer, speedの前処理を行うクラス
        :param range_max: Lidarの最大距離 (m)
        :param base_num: 元のLidarスキャンのサンプル数
        :param downsample_num: ダウンサンプリング後のサンプル数

        例えば、base_num=1081, downsample_num=181ならば、元のLidarスキャンを181個にダウンサンプリングする
        URG_Nodeから受けとるLidarスキャンは1081個のサンプルを持つ
        """
        self.range_max = range_max
        self.base_num = base_num
        self.downsample_num = downsample_num
        self.sample_indices = np.round(np.linspace(0, base_num - 1, downsample_num)).astype(int)

    def __call__(self, scan, steer, speed):
        # scan 正規化 + ダウンサンプリング
        scan = np.clip(scan, 0, self.range_max) / self.range_max
        scan = scan[self.sample_indices]

        # steer / speed のクリップ
        steer = np.clip(steer, -1.0, 1.0)
        speed = np.clip(speed, -1.0, 1.0)

        return scan, steer, speed

class E2ESeqTransform:
    """
    【時系列対応版】
    Lidar、steer、speedのシーケンスデータに対して前処理を行うクラス。
    """
    def __init__(self, range_max=30.0, base_num=1081, downsample_num=181):
        """
        :param range_max: Lidarの最大距離 (m)
        :param base_num: 元のLidarスキャンのサンプル数
        :param downsample_num: ダウンサンプリング後のサンプル数
        """
        self.range_max = range_max
        self.base_num = base_num
        self.downsample_num = downsample_num
        # ダウンサンプリングに使うインデックスの計算は変更なし
        self.sample_indices = np.round(np.linspace(0, base_num - 1, downsample_num)).astype(int)

    def __call__(self, scan_seq, steer_seq, speed_seq):
        """
        シーケンスデータを受け取り、まとめて前処理を行う。
        :param scan_seq: (sequence_length, base_num) の形状を持つNumpy配列
        :param steer_seq: (sequence_length,) の形状を持つNumpy配列
        :param speed_seq: (sequence_length,) の形状を持つNumpy配列
        """
        # scan_seq の正規化 + ダウンサンプリング
        # 2次元配列に対してもNumPyの演算は効率的に機能する
        scan_seq = np.clip(scan_seq, 0, self.range_max) / self.range_max
        # すべてのタイムステップ(行)に対して、同じインデックス(列)でダウンサンプリング
        scan_seq = scan_seq[:, self.sample_indices]

        # steer_seq / speed_seq のクリップ
        # 1次元配列に対しても同様に機能する
        steer_seq = np.clip(steer_seq, -1.0, 1.0)
        speed_seq = np.clip(speed_seq, -1.0, 1.0)

        return scan_seq, steer_seq, speed_seq