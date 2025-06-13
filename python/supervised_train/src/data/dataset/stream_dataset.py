import os
import numpy as np
import torch
from torch.utils.data import IterableDataset
import random

class StreamDataset(IterableDataset):
    """
    ステートフルな逐次学習のためのデータセット（修正版）。

    batch_sizeの数だけ独立したデータストリームを作成する。
    各ストリームは、それぞれが独立したランダムな順序で、"全ての"bagを処理する。
    """
    def __init__(self, root_dir, batch_size, sequence_length, transform=None):
        """
        Args:
            root_dir (string): .npyファイル群が格納されたルートディレクトリ。
            batch_size (int): 並列で処理するシーケンス（ストリーム）の数。
            sequence_length (int): 1サンプルとして切り出すシーケンスの長さ。
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
            
        self.root_dir = os.path.expanduser(root_dir)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.transform = transform
        
        self.bags_data = self._load_data()

        if not self.bags_data:
            raise ValueError("No valid bags were found in the root directory.")

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
            # __getitem__のprev_action_seqのために+1が必要
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
        
        # 1. ★★★ここが変更点★★★
        # 各ワーカー（ストリーム）が、それぞれ独立した順序で全bagを処理するキューを作成する
        worker_queues = []
        all_bag_indices = list(range(len(self.bags_data)))
        
        for _ in range(self.batch_size):
            # ループのたびに、全bagのインデックスリストを新たにシャッフルする
            shuffled_indices = all_bag_indices[:] # スライスでコピーを作成
            random.shuffle(shuffled_indices)
            worker_queues.append(shuffled_indices)

        # 2. 各ワーカーの状態を管理するカーソルを初期化 (変更なし)
        worker_bag_cursor = [0] * self.batch_size 
        worker_frame_cursor = [0] * self.batch_size

        N = self.sequence_length

        # 3. すべてのワーカーが全タスクを終えるまでループ (変更なし)
        while True:
            batch = {
                'scan_seq': [],
                'prev_action_seq': [],
                'target_action_seq': [],
                'is_first_seq': []
            }
            active_workers = 0

            # 4. 各ワーカーからデータを1つずつ集めてバッチを作成 (変更なし)
            for i in range(self.batch_size):
                if worker_bag_cursor[i] >= len(worker_queues[i]):
                    continue

                current_bag_queue_idx = worker_bag_cursor[i]
                bag_idx = worker_queues[i][current_bag_queue_idx]
                bag_data = self.bags_data[bag_idx]
                
                start_frame = worker_frame_cursor[i]

                # このbagの残りがsequence_lengthより短い場合、このbagは終了
                # prev_action_seq のために len(scans) より厳密には len(actions)で判定
                if start_frame + N >= len(bag_data['actions']):
                    worker_bag_cursor[i] += 1
                    worker_frame_cursor[i] = 0
                    continue

                active_workers += 1
                
                sample = {
                    'scan_seq': bag_data['scans'][start_frame : start_frame + N],
                    'prev_action_seq': bag_data['actions'][start_frame - 1 : start_frame + N - 1],
                    'target_action_seq': bag_data['actions'][start_frame : start_frame + N]
                }
                
                is_first = (start_frame == 0)

                batch['scan_seq'].append(sample['scan_seq'])
                batch['prev_action_seq'].append(sample['prev_action_seq'])
                batch['target_action_seq'].append(sample['target_action_seq'])
                batch['is_first_seq'].append(is_first)

                worker_frame_cursor[i] += N

            if active_workers == 0:
                break

            if active_workers > 0:
                final_batch = {
                    'scan_seq': torch.from_numpy(np.array(batch['scan_seq'], dtype=np.float32)),
                    'prev_action_seq': torch.from_numpy(np.array(batch['prev_action_seq'], dtype=np.float32)),
                    'target_action_seq': torch.from_numpy(np.array(batch['target_action_seq'], dtype=np.float32)),
                    'is_first_seq': torch.tensor(batch['is_first_seq'], dtype=torch.bool)
                }
                yield final_batch