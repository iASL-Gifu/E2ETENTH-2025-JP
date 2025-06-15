import os
import numpy as np
import torch
from torch.utils.data import IterableDataset
import random
import math

from .transform import StreamAugmentor 
class StreamDataset(IterableDataset):
    """
    ステートフルな逐次学習のためのデータセット。
    StreamAugmentorと連携してエピソード単位のデータ拡張を行う。
    """
    def __init__(self, root_dir, batch_size, sequence_length, augmentor: StreamAugmentor = None, transform=None):
        """
        Args:
            root_dir (string): .npyファイル群が格納されたルートディレクトリ。
            batch_size (int): 並列で処理するシーケンス（ストリーム）の数。
            sequence_length (int): 1サンプルとして切り出すシーケンスの長さ。
            augmentor (StreamAugmentor, optional): エピソード単位の拡張を行うクラス。
            transform (callable, optional): Augmentorによる拡張後に適用される変換（正規化、Tensor化など）。
        """
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
            
        self.root_dir = os.path.expanduser(root_dir)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.augmentor = augmentor # ★★★ Augmentorを受け取る
        self.transform = transform
        
        self.bags_data = self._load_data()

        if not self.bags_data:
            raise ValueError("No valid bags were found in the root directory.")
        
    def __len__(self):
        total_sequences = 0
        for bag_data in self.bags_data:
            # 例: actionsが100フレーム、sequence_lengthが10なら、91個のシーケンス (0-9, 1-10, ..., 90-99)
            num_frames_in_bag = len(bag_data['actions'])
            if num_frames_in_bag >= self.sequence_length:
                total_sequences += (num_frames_in_bag - self.sequence_length + 1) // self.sequence_length

        if self.batch_size == 0: # 念のため0除算を避ける
            return total_sequences # batch_sizeが0は想定外だが、安全のため

        return max(1, math.ceil(total_sequences / self.batch_size))

    def _load_data(self):
        """ルートディレクトリからすべてのbagデータをメモリに読み込む"""
        bags_data = []
        for bag_name in sorted(os.listdir(self.root_dir)):
            bag_dir = os.path.join(self.root_dir, bag_name)
            if not os.path.isdir(bag_dir):
                continue

            scans_path = os.path.join(bag_dir, 'scans.npy')
            steers_path = os.path.join(bag_dir, 'steers.npy')
            speeds_path = os.path.join(bag_dir, 'speeds.npy')
            if not all(os.path.exists(p) for p in [scans_path, steers_path, speeds_path]):
                continue
            
            scans = np.load(scans_path)
            # 少なくとも1つのシーケンスが作れる長さが必要
            if len(scans) < self.sequence_length + 1:
                continue

            steers = np.load(steers_path)
            speeds = np.load(speeds_path)
            actions = np.stack([steers, speeds], axis=1).astype(np.float32)
            
            bags_data.append({'scans': scans, 'actions': actions})
        
        print(f"[INFO] Loaded {len(bags_data)} bags for stateful iteration.")
        return bags_data

    def __iter__(self):
        """データセットのイテレータを返す。"""
        
        # 1. ★★★ エピソードごとの拡張計画を立てる ★★★
        episode_plans = {}
        if self.augmentor:
            for i in range(len(self.bags_data)):
                episode_plans[i] = self.augmentor.plan_for_episode()

        # 2. 各ワーカー（ストリーム）のキューを作成
        worker_queues = []
        all_bag_indices = list(range(len(self.bags_data)))
        for _ in range(self.batch_size):
            shuffled_indices = all_bag_indices[:]
            random.shuffle(shuffled_indices)
            worker_queues.append(shuffled_indices)

        # 3. 各ワーカーの状態を管理するカーソルを初期化
        worker_bag_cursor = [0] * self.batch_size 
        worker_frame_cursor = [0] * self.batch_size
        N = self.sequence_length

        # 4. すべてのワーカーが全タスクを終えるまでループ
        while True:
            batch = {
                'scan_seq': [], 'prev_action_seq': [], 'target_action_seq': [], 'is_first_seq': []
            }
            active_workers = 0
            
            # 各ワーカーの元の状態を保持
            initial_worker_frame_cursors = list(worker_frame_cursor)
            initial_worker_bag_cursors = list(worker_bag_cursor)


            # 5. 各ワーカーからデータを1つずつ集めてバッチを作成
            for i in range(self.batch_size):
                if worker_bag_cursor[i] >= len(worker_queues[i]):
                    # このワーカーはすでに終了している
                    # ★★★ Padding用のダミーサンプルを追加 ★★★
                    # ダミーサンプルの形状を元のデータに合わせる必要があります。
                    # ここでは、最初のbag_dataの形状を参考にします。
                    if not self.bags_data: # データがない場合はスキップ
                        continue

                    # ダミーデータの形状を決定するための参照
                    ref_scan_shape = (N,) + self.bags_data[0]['scans'].shape[1:]
                    ref_action_shape = (N,) + self.bags_data[0]['actions'].shape[1:]

                    batch['scan_seq'].append(np.zeros(ref_scan_shape, dtype=np.float32))
                    batch['prev_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                    batch['target_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                    batch['is_first_seq'].append(True) # ダミーなので常にリセット
                    continue

                current_bag_queue_idx = worker_bag_cursor[i]
                bag_idx = worker_queues[i][current_bag_queue_idx]
                bag_data = self.bags_data[bag_idx]
                
                start_frame = worker_frame_cursor[i]

                if start_frame + N >= len(bag_data['actions']):
                    # このワーカーの現在のbagが終了した
                    worker_bag_cursor[i] += 1
                    worker_frame_cursor[i] = 0
                    
                    # ★★★ Padding用のダミーサンプルを追加 ★★★
                    if not self.bags_data: # データがない場合はスキップ
                        continue

                    ref_scan_shape = (N,) + self.bags_data[0]['scans'].shape[1:]
                    ref_action_shape = (N,) + self.bags_data[0]['actions'].shape[1:]

                    batch['scan_seq'].append(np.zeros(ref_scan_shape, dtype=np.float32))
                    batch['prev_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                    batch['target_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                    batch['is_first_seq'].append(True) # ダミーなので常にリセット
                    continue

                active_workers += 1
            
                # 元データをNumpyとして取得
                current_scan_seq = bag_data['scans'][start_frame : start_frame + N]
                current_target_action_seq = bag_data['actions'][start_frame : start_frame + N]

                if start_frame == 0:
                    dummy_prev_action = np.zeros_like(bag_data['actions'][0])
                    current_prev_action_seq = np.concatenate(
                        (dummy_prev_action[np.newaxis, :], bag_data['actions'][start_frame : start_frame + N - 1]),
                        axis=0
                    )
                else:
                    current_prev_action_seq = bag_data['actions'][start_frame - 1 : start_frame + N - 1]

                sample = {
                    'scan_seq': current_scan_seq,
                    'prev_action_seq': current_prev_action_seq.copy(),
                    'target_action_seq': current_target_action_seq.copy()
                }
                
                if self.augmentor and bag_idx in episode_plans:
                    plan = episode_plans[bag_idx]
                    sample = self.augmentor.apply(sample, plan)

                if self.transform:
                    sample = self.transform(sample)

                is_first = (start_frame == 0)

                for key in ['scan_seq', 'prev_action_seq', 'target_action_seq']:
                    batch[key].append(sample[key])
                batch['is_first_seq'].append(is_first)

                worker_frame_cursor[i] += N

            # ループを終了する条件を再評価
            # 全ワーカーのカーソルが進んでいない場合、つまりデータが全く供給されない場合は終了
            if all(worker_frame_cursor[i] == initial_worker_frame_cursors[i] and \
                   worker_bag_cursor[i] == initial_worker_bag_cursors[i] for i in range(self.batch_size)):
                break

            # active_workers は Padding 後の処理では使われないので削除
            # if active_workers > 0: # この条件は不要になり、常に yield する
            # バッチ内のデータをNumpy配列にまとめ、Torch Tensorに変換
            final_batch = {
                key: torch.from_numpy(np.array(val, dtype=np.float32))
                for key, val in batch.items() if key != 'is_first_seq'
            }
            final_batch['is_first_seq'] = torch.tensor(batch['is_first_seq'], dtype=torch.bool)
            
            yield final_batch