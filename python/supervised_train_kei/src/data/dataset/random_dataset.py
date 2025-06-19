import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

class SequenceRndDataset(Dataset):
    """
    単一のルートディレクトリからシーケンス対シーケンス学習のためのサンプルを生成するデータセット。
    """
    def __init__(self, seq_dir, sequence_length=10, transform=None):
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
        
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples_info = [] # (start_idx,) のタプルのリスト

        # データのロード
        scans_path = os.path.join(seq_dir, 'scans.npy')
        steers_path = os.path.join(seq_dir, 'steers.npy')
        speeds_path = os.path.join(seq_dir, 'speeds.npy')
        
        if not all(os.path.exists(p) for p in [scans_path, steers_path, speeds_path]):
            raise FileNotFoundError(f"Missing one or more .npy files in {seq_dir}")
        
        self.scans = np.load(scans_path)
        if len(self.scans) < self.sequence_length + 1:
            # シーケンス長+1 に満たない場合は、このバッグからはサンプルを生成しない
            print(f"Warning: Bag {seq_dir} too short for sequence_length {sequence_length}. Skipping.")
            return
        
        steers = np.load(steers_path)
        speeds = np.load(speeds_path)
        self.actions = np.stack([steers, speeds], axis=1).astype(np.float32)
        
        num_frames_in_bag = len(self.scans)
        
        # シーケンスの開始インデックスを生成
        for i in range(1, num_frames_in_bag - self.sequence_length + 1, self.sequence_length):
            self.samples_info.append(i) # ここでは開始インデックスのみを格納

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, idx):
        start_idx = self.samples_info[idx]
        N = self.sequence_length
        
        sample = {
            'scan_seq': self.scans[start_idx : start_idx + N],
            'prev_action_seq': self.actions[start_idx - 1 : start_idx + N - 1],
            'target_action_seq': self.actions[start_idx : start_idx + N]
        }

        if self.transform:
            sample = self.transform(sample)

        return {
            'scan_seq': torch.from_numpy(sample['scan_seq'].astype(np.float32)),
            'prev_action_seq': torch.from_numpy(sample['prev_action_seq']),
            'target_action_seq': torch.from_numpy(sample['target_action_seq']),
            'is_first_seq': torch.tensor(True, dtype=torch.bool) 
        }

class ConcatRndDataset(Dataset):
    """
    複数のSingleSequenceDatasetインスタンスを結合するデータセット。
    指定されたroot_dirを再帰的に探索し、条件を満たすディレクトリからデータを読み込む。
    """
    def __init__(self, root_dir, sequence_length=10, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        
        self.datasets = []
        root_dir = os.path.expanduser(root_dir)

        print(f"[INFO] Recursively searching for sequence data in '{root_dir}'...")

        # os.walkを使ってディレクトリを再帰的に探索
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 必須ファイルが現在のディレクトリに存在するかチェック
            required_files = ['scans.npy', 'steers.npy', 'speeds.npy']
            if all(f in filenames for f in required_files):
                print(f"[INFO] Found valid sequence data at: {dirpath}")
                try:
                    # 条件を満たしたディレクトリでSequenceRndDatasetを作成
                    single_dataset = SequenceRndDataset(dirpath, sequence_length, transform)
                    if len(single_dataset) > 0: # サンプルが生成された場合のみ追加
                        self.datasets.append(single_dataset)
                    else:
                        print(f"[INFO] Skipping {dirpath} because it's too short to create any samples.")
                except Exception as e:
                    print(f"Skipping {dirpath} due to an error: {e}")

        # PyTorchのConcatDatasetを利用して、複数のデータセットを結合
        if not self.datasets:
            print("[INFO] No valid datasets found to combine.")
            self.combined_dataset = [] # エラーを防ぐために空リストをセット
        else:
            self.combined_dataset = ConcatDataset(self.datasets)

        print("-" * 50)
        print(f"[INFO] Created a total of {len(self.combined_dataset)} samples from {len(self.datasets)} sequence directories.")
        if len(self.datasets) > 0:
            print("[INFO] Remainder blocks shorter than sequence_length were discarded.")
        print("-" * 50)


    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]