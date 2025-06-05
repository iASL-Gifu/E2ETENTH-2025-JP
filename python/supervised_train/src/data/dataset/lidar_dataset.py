import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LidarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        root_dir = os.path.expanduser(root_dir)

        for bag_name in os.listdir(root_dir):
            bag_dir = os.path.join(root_dir, bag_name)
            if not os.path.isdir(bag_dir):
                continue

            scans_path = os.path.join(bag_dir, 'scans.npy')
            steers_path = os.path.join(bag_dir, 'steers.npy')
            speeds_path = os.path.join(bag_dir, 'speeds.npy')

            if not (os.path.exists(scans_path) and os.path.exists(steers_path) and os.path.exists(speeds_path)):
                print(f"[WARN] Skipping incomplete bag: {bag_name}")
                continue

            scans = np.load(scans_path)
            steers = np.load(steers_path)
            speeds = np.load(speeds_path)

            for i in range(len(scans)):
                self.samples.append((scans[i], steers[i], speeds[i]))

        print(f"[INFO] Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scan, steer, speed = self.samples[idx]

        if self.transform:
            scan, steer, speed = self.transform(scan, steer, speed)

        return {
            'scan': torch.from_numpy(scan.astype(np.float32)),
            'steer': torch.tensor(steer, dtype=torch.float32),
            'speed': torch.tensor(speed, dtype=torch.float32)
        }


class LidarSeqDataset(Dataset):
    """
    時系列データを【重複しないシーケンス単位】で取得するデータセットクラス。
    シーケンス長に満たない余りのデータは切り捨てられる。
    """
    def __init__(self, root_dir, sequence_length=10, transform=None):
        """
        Args:
            root_dir (string): データセットのルートディレクトリ。
            sequence_length (int): 1つのサンプルとして取得するシーケンスの長さ。
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """
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

            if not (os.path.exists(scans_path) and os.path.exists(steers_path) and os.path.exists(speeds_path)):
                print(f"[WARN] Skipping incomplete bag: {bag_name}")
                continue
            
            scans = np.load(scans_path)
            steers = np.load(steers_path)
            speeds = np.load(speeds_path)

            current_bag_index = len(self.bags_data)
            self.bags_data.append({'scans': scans, 'steers': steers, 'speeds': speeds})

            num_frames_in_bag = len(scans)
            
            end_point = (num_frames_in_bag // self.sequence_length) * self.sequence_length
            for i in range(0, end_point, self.sequence_length):
                self.samples_info.append((current_bag_index, i))
            # -----------------------------------------------------------

        print(f"[INFO] Loaded {len(self.bags_data)} bags.")
        print(f"[INFO] Created {len(self.samples_info)} non-overlapping samples with sequence length {self.sequence_length}.")

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, idx):
        bag_index, start_index = self.samples_info[idx]
        end_index = start_index + self.sequence_length
        bag_data = self.bags_data[bag_index]

        # まずNumPy配列としてシーケンスデータを取得
        scan_seq = bag_data['scans'][start_index:end_index]
        steer_seq = bag_data['steers'][start_index:end_index]
        speed_seq = bag_data['speeds'][start_index:end_index]

        # --- 【ここが連携部分】 ---
        # transformが指定されていれば、シーケンスデータに適用する
        if self.transform:
            scan_seq, steer_seq, speed_seq = self.transform(scan_seq, steer_seq, speed_seq)
        # -------------------------

        # 最後にPyTorchテンソルに変換して返す
        return {
            'scan_seq': torch.from_numpy(scan_seq.astype(np.float32)),
            'steer_seq': torch.from_numpy(steer_seq.astype(np.float32)),
            'speed_seq': torch.from_numpy(speed_seq.astype(np.float32))
        }