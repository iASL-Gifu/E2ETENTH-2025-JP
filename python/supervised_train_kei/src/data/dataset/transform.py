import numpy as np
import random

class SeqToSeqTransform:
    """
    LidarSeqToSeqDataset用のデータ前処理クラス。
    ベース処理として正規化・ダウンサンプリング・クリッピングを行い、
    データ拡張として左右反転とノイズ付与を確率的に適用する。
    """
    def __init__(self,
                 # --- ベース処理パラメータ ---
                 range_max: float = 30.0,
                 base_num: int = 1080,
                 downsample_num: int = 100,
                 # --- データ拡張パラメータ ---
                 augment: bool = True,
                 flip_prob: float = 0.5,
                 noise_std: float = 0.01):
        """
        :param range_max: LiDARの最大距離 (m)
        :param base_num: 元のLiDARスキャンのサンプル数
        :param downsample_num: ダウンサンプリング後のサンプル数
        :param augment: データ拡張を行うかどうかのフラグ
        :param flip_prob: 左右反転を適用する確率
        :param noise_std: 追加するガウスノイズの標準偏差 (0にすると適用されない)
        """
        # ベース処理用の設定
        self.range_max = range_max
        self.sample_indices = np.round(np.linspace(0, base_num - 1, downsample_num)).astype(int)

        # データ拡張用の設定
        self.augment = augment
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        
        

    def downsample_single_frame(self, scan_data: np.ndarray, target_size: int, front_ratio: float) -> np.ndarray:
        """
        1つの2D LiDARフレーム（1080ビーム、-135°～+135°）を適応的ダウンサンプリング
        
        Args:
            scan_data: LiDARスキャンデータ（1D配列、通常1080要素）
            target_size: 目標サンプル数
        
        Returns:
            ダウンサンプリングされたデータ
        
        例:
            # 1080ビーム（-135°～+135°）のLiDARデータを100点にダウンサンプリング
            original_scan = np.array([...])  # 1080個の距離値
            downsampled = downsample_single_frame(original_scan, 100)  # 100点（前方70点、側面30点）
            # 前方±30度エリア（240ビーム）が高密度、側面エリア（840ビーム）が低密度でサンプリングされる
        """
        # print(scan_data.shape)
        scan_data = scan_data[0]
        if target_size is None or scan_data.size == target_size:
            return scan_data
        
        # 2D LiDARスキャンの角度配列を生成（-135°から+135°の範囲）
        start_angle = -135 * np.pi / 180  # -135度をラジアンに変換
        end_angle = 135 * np.pi / 180     # +135度をラジアンに変換
        angles = np.linspace(start_angle, end_angle, scan_data.size)
        
        # 前方セクターの定義（前方向から±30度）
        front_angle_range = np.pi / 6  # 30度をラジアンで
        is_front = np.abs(angles) <= front_angle_range
        
        # サンプリング比率の計算
        # 1080ビーム中、前方240ビーム（±30度）に重点を置く
        front_ratio = front_ratio  # 前方エリアに70%のサンプル
        
        front_count = int(target_size * front_ratio)
        side_count = target_size - front_count
        
        # 前方エリアと側面エリアのインデックスを取得
        front_indices = np.where(is_front)[0]
        side_indices = np.where(~is_front)[0]
        
        selected_indices = []
        
        # 前方エリアを高密度でサンプリング
        if len(front_indices) > 0:
            if front_count >= len(front_indices):
                selected_indices.extend(front_indices)
                remaining = front_count - len(front_indices)
                side_count += remaining
            else:
                front_sample_indices = np.linspace(0, len(front_indices) - 1, front_count, dtype=int)
                selected_indices.extend(front_indices[front_sample_indices])
        
        # 側面エリアを低密度でサンプリング
        if len(side_indices) > 0 and side_count > 0:
            if side_count >= len(side_indices):
                selected_indices.extend(side_indices)
            else:
                side_sample_indices = np.linspace(0, len(side_indices) - 1, side_count, dtype=int)
                selected_indices.extend(side_indices[side_sample_indices])
        
        # 元の順序を維持するためにインデックスをソート
        selected_indices = np.sort(selected_indices)
        
        return scan_data[selected_indices]

    def __call__(self, sample: dict) -> dict:
        """
        データセットから取得したサンプルに前処理とデータ拡張を適用する。
        入力された辞書 `sample` を直接更新し、その `sample` を返す。
        """
        scan_seq = sample['scan_seq']
        if scan_seq.shape[1] == 1081:
            scan_seq = scan_seq[:, :-1]  # 最後の点(1081番目)を除外し、1080点にする
        
        # --- 1. ベースの前処理 ---
        # scan_seq を直接更新
        sample['scan_seq'] = self.downsample_single_frame(sample["scan_seq"],100,0.7)

        # prev_action_seq と target_action_seq は参照渡しになるので、
        # 必要に応じて `.copy()` で新しい配列を作成し、
        # 変換後に元の辞書キーを新しい配列で上書きします。
        # データ拡張部分で内容が変更されるので、ここでコピーしておくと安全です。
        # prev_action_seq_copy = sample['prev_action_seq'].copy()
        # target_action_seq_copy = sample['target_action_seq'].copy()

        # # --- 2. データ拡張 (augmentフラグがTrueの場合のみ実行) ---
        # if self.augment:
        #     # 2-1. 左右反転
        #     if random.random() < self.flip_prob:
        #         sample['scan_seq'] = np.flip(sample['scan_seq'], axis=1) # 直接更新
                
        #         # steer ([:, 0]) の符号を反転
        #         prev_action_seq_copy[:, 0] *= -1
        #         target_action_seq_copy[:, 0] *= -1

        #     # 2-2. ノイズ付与
        #     if self.noise_std > 0:
        #         # ガウスノイズを生成
        #         noise = np.random.normal(0, self.noise_std, sample['scan_seq'].shape)
        #         # ノイズを付与し、値が [0, 1] の範囲に収まるようにクリップ
        #         sample['scan_seq'] = np.clip(sample['scan_seq'] + noise, 0, 1.0) # 直接更新

        # # --- 3. アクションのクリッピング (最終処理) ---
        # # 拡張処理で値が範囲外に出る可能性も考慮し、最後にクリッピングを行う
        # np.clip(prev_action_seq_copy[:, 0], -1.0, 1.0, out=prev_action_seq_copy[:, 0]) # steer
        # np.clip(prev_action_seq_copy[:, 1], -1.0, 1.0, out=prev_action_seq_copy[:, 1]) # speed
        # np.clip(target_action_seq_copy[:, 0], -1.0, 1.0, out=target_action_seq_copy[:, 0]) # steer
        # np.clip(target_action_seq_copy[:, 1], -1.0, 1.0, out=target_action_seq_copy[:, 1]) # speed

        # # 更新されたアクション配列を元の辞書に戻す
        # sample['prev_action_seq'] = prev_action_seq_copy
        # sample['target_action_seq'] = target_action_seq_copy

        # 元の sample 辞書を直接返します。
        # これにより、'is_first_seq' や他のキーが変更されずに保持されます。
        return sample

    

class StreamAugmentor:
    """
    ストリーミングデータセット用の、エピソード単位のデータ拡張を管理・適用するクラス。
    """
    def __init__(self, augment=True, flip_prob=0.5, noise_std=0.01):
        """
        Args:
            augment (bool): データ拡張を有効にするか。
            flip_prob (float): 左右反転を適用する確率。
            noise_std (float): 付与するガウスノイズの標準偏差。
        """
        self.augment = augment
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        # ここに将来的な拡張（例: rotation_probなど）を追加できる

    def plan_for_episode(self):
        """
        1つのエピソード（bag）に対する拡張計画をランダムに立てる。

        Returns:
            dict: このエピソードに適用する拡張内容を記述した辞書。
        """
        if not self.augment:
            return {'flip': False, 'apply_noise': False}

        plan = {
            'flip': random.random() < self.flip_prob,
            'apply_noise': self.noise_std > 0 and random.random() < 0.5 # 例: 50%の確率でノイズを適用
        }
        return plan

    def apply(self, sample, plan):
        """
        与えられたサンプルに、計画に基づいてデータ拡張を適用する。
        入力と出力はnumpy配列の辞書。

        Args:
            sample (dict): 拡張前のデータサンプル。
            plan (dict): plan_for_episode()で生成された計画。

        Returns:
            dict: 拡張が適用されたデータサンプル。
        """
        # --- 左右反転 ---
        if plan.get('flip', False):
            # scan_seqの点群方向(axis=1)を反転
            sample['scan_seq'] = np.flip(sample['scan_seq'], axis=1)
            # actionのsteer([:, 0])の符号を反転
            sample['prev_action_seq'][:, 0] *= -1
            sample['target_action_seq'][:, 0] *= -1

        # --- ノイズ付与 ---
        if plan.get('apply_noise', False):
            noise = np.random.normal(0, self.noise_std, sample['scan_seq'].shape)
            # scan_seqは事前に正規化されていると仮定し、[0, 1]の範囲にクリップ
            sample['scan_seq'] = np.clip(sample['scan_seq'] + noise, 0, 1.0)

        return sample