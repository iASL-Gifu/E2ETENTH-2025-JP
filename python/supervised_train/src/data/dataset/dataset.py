import torch
from torch.utils.data import DataLoader
import math

from .random_dataset import RandomDataset
from .stream_dataset import StreamDataset

class HybridLoader:
    """
    ランダムサンプリングとストリームサンプリングを統合するハイブリッドデータローダ。（修正版）
    """
    def __init__(self, root_dir, sequence_length, total_batch_size, random_ratio=0.5, transform=None, num_workers_random=4):
        """
        Args:
            root_dir (string): データセットのルートディレクトリ。
            sequence_length (int): シーケンスの長さ。
            total_batch_size (int): 1バッチあたりの合計サンプル数。
            random_ratio (float): 全バッチサイズのうち、ランダムサンプルが占める割合 (0.0 ~ 1.0)。
            transform (callable, optional): サンプルに適用されるオプションの変換。 ### ★修正点★ ###
            num_workers_random (int): ランダムデータローダで使用するワーカー数。
        """
        if not (0.0 <= random_ratio <= 1.0):
            raise ValueError("random_ratio must be between 0.0 and 1.0.")

        # 1. 各ローダのバッチサイズを計算
        random_batch_size = math.ceil(total_batch_size * random_ratio)
        stream_batch_size = total_batch_size - random_batch_size

        print(f"[INFO] HybridLoader initialized.")
        print(f"  Total batch size: {total_batch_size}")
        print(f"  - Random samples (BPTT): {random_batch_size} (Ratio: {random_ratio:.2f})")
        print(f"  - Stream samples (TBPTT): {stream_batch_size} (Ratio: {1-random_ratio:.2f})")

        self.random_batch_size = random_batch_size
        self.stream_batch_size = stream_batch_size

        # 2. 各データセットとデータローダを初期化
        if random_batch_size > 0:
            # ### ★修正点★ ### transformを渡す
            random_dataset = RandomDataset(root_dir, sequence_length, transform=transform)
            self.random_loader = DataLoader(
                random_dataset,
                batch_size=random_batch_size,
                shuffle=True,
                num_workers=num_workers_random,
                drop_last=True
            )
        else:
            self.random_loader = None

        if stream_batch_size > 0:
            # ### ★修正点★ ### transformを渡す
            stream_dataset = StreamDataset(root_dir, stream_batch_size, sequence_length, transform=transform)
            self.stream_loader = DataLoader(
                stream_dataset,
                batch_size=None,
                num_workers=0
            )
        else:
            self.stream_loader = None
            
        if self.random_loader and self.stream_loader:
            self.loader_iterator = zip(self.random_loader, self.stream_loader)
        elif self.random_loader:
            self.loader_iterator = self.random_loader
        elif self.stream_loader:
            self.loader_iterator = self.stream_loader
        else:
            raise ValueError("Both random_batch_size and stream_batch_size are zero.")

    def __iter__(self):
        # イテレーションが始まるたびに、イテレータを再生成する
        if self.random_loader and self.stream_loader:
            self.loader_iterator = zip(self.random_loader, self.stream_loader)
        
        for batch_parts in self.loader_iterator:
            if self.random_loader and self.stream_loader:
                random_batch, stream_batch = batch_parts
                combined_batch = {
                    key: torch.cat([random_batch[key], stream_batch[key]], dim=0)
                    for key in random_batch.keys()
                }
            elif self.random_loader:
                 combined_batch = batch_parts
            else:
                 combined_batch = batch_parts

            yield combined_batch

    def __len__(self):
        if self.random_loader:
            return len(self.random_loader)
        else:
            print("Warning: Length of StreamDataset is undefined.")
            return 0