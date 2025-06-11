import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LidarSeqToSeqDataset(Dataset):
    """
    シーケンス対シーケンス学習のためのデータセット。
    各タイムステップで逐次的にアクションを推論し、損失を計算するモデル向け。
    """
    def __init__(self, root_dir, sequence_length=10, transform=None):
        """
        Args:
            root_dir (string): .npyファイル群が格納されたルートディレクトリ。
            sequence_length (int): 1つのサンプルとして切り出すシーケンスの長さ。
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
            
        self.sequence_length = sequence_length
        self.transform = transform
        
        self.bags_data = []
        self.samples_info = []

        root_dir = os.path.expanduser(root_dir)

        for bag_name in sorted(os.listdir(root_dir)):
            
            bag_dir = os.path.join(root_dir, bag_name)
            if not os.path.isdir(bag_dir):
                continue

            scans_path = os.path.join(bag_dir, 'scans.npy')
            steers_path = os.path.join(bag_dir, 'steers.npy')
            speeds_path = os.path.join(bag_dir, 'speeds.npy')
            if not all(os.path.exists(p) for p in [scans_path, steers_path, speeds_path]):
                continue
            
            scans = np.load(scans_path)
            # 各ステップでprev_actionを必要とするため、sequence_length + 1 以上の長さが必要
            if len(scans) < self.sequence_length + 1:
                continue

            steers = np.load(steers_path)
            speeds = np.load(speeds_path)
            actions = np.stack([steers, speeds], axis=1).astype(np.float32)
            
            current_bag_index = len(self.bags_data)
            self.bags_data.append({'scans': scans, 'actions': actions})

            num_frames_in_bag = len(scans)
            # スライディングウィンドウでサンプル情報を作成
            # prev_actionを安全に取得するため、ループの開始位置に注意
            for i in range(1, num_frames_in_bag - self.sequence_length + 1):
                # i は prev_action_seq の開始インデックス
                self.samples_info.append((current_bag_index, i))

        print(f"[INFO] Created {len(self.samples_info)} sequence-to-sequence samples with length {self.sequence_length}.")

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, idx):
        bag_index, start_idx = self.samples_info[idx]
        N = self.sequence_length
        
        bag_data = self.bags_data[bag_index]
        scans = bag_data['scans']
        actions = bag_data['actions']
        
        sample = {
            'scan_seq': scans[start_idx : start_idx + N],
            'prev_action_seq': actions[start_idx - 1 : start_idx + N - 1],
            'target_action_seq': actions[start_idx : start_idx + N]
        }

        if self.transform:
            sample = self.transform(sample)

        return {
            'scan_seq': torch.from_numpy(sample['scan_seq'].astype(np.float32)),
            'prev_action_seq': torch.from_numpy(sample['prev_action_seq']),
            'target_action_seq': torch.from_numpy(sample['target_action_seq'])
        }