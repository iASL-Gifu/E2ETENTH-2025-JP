import torch
from torch.utils.data import DataLoader
import math

# --- 連携するクラスのインポートを想定 ---
from .random_dataset import RandomDataset
from .stream_dataset import StreamDataset
from .transform import StreamAugmentor # 例

class HybridLoader:
    """
    ランダムサンプリングとストリームサンプリングを統合するハイブリッドデータローダ。
    各データセットに合わせたデータ変換・拡張処理を個別に設定できる。
    """
    def __init__(self,
                 root_dir,
                 sequence_length,
                 total_batch_size,
                 random_ratio=0.5,
                 transform_random=None,
                 transform_stream=None,
                 augmentor_stream=None,
                 num_workers_random=4):
        """
        Args:
            root_dir (string): データセットのルートディレクトリ。
            sequence_length (int): シーケンスの長さ。
            total_batch_size (int): 1バッチあたりの合計サンプル数。
            random_ratio (float): 全バッチサイズのうち、ランダムサンプルが占める割合 (0.0 ~ 1.0)。
            transform_random (callable, optional): RandomDatasetに適用する変換（拡張＋ベース変換）。
            transform_stream (callable, optional): StreamDatasetに適用する変換（ベース変換のみ）。
            augmentor_stream (StreamAugmentor, optional): StreamDatasetに適用するエピソード単位の拡張。
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
            # RandomDatasetには、従来のデータ拡張とベース変換を含むtransformを渡す
            random_dataset = RandomDataset(root_dir, sequence_length, transform=transform_random)
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
            # StreamDatasetには、augmentorとベース変換のみのtransformを渡す
            stream_dataset = StreamDataset(
                root_dir,
                stream_batch_size,
                sequence_length,
                augmentor=augmentor_stream,
                transform=transform_stream
            )
            self.stream_loader = DataLoader(
                stream_dataset,
                batch_size=None, # Dataset内部でバッチングするためNone
                num_workers=0,      # IterableDatasetのマルチプロセスは複雑なので0を推奨
                drop_last=False,     # 最後の不完全なバッチをドロップ
            )
        else:
            self.stream_loader = None
            
        # 3. イテレータの準備 (変更なし)
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
                # 2つのバッチを結合
                combined_batch = {
                    key: torch.cat([random_batch[key], stream_batch[key]], dim=0)
                    for key in random_batch.keys()
                }
            elif self.random_loader:
                 combined_batch = batch_parts
            else: # stream_loaderのみ
                 combined_batch = batch_parts

            yield combined_batch

    def __len__(self):
        # 長さは有限のRandomLoaderに依存させるのが一般的
        if self.random_loader:
            return len(self.random_loader)
        elif self.stream_loader:
            # StreamDatasetの長さは定義できないため、警告を出すか0を返す
            print("Warning: Length of HybridLoader is undefined because it only contains a stream component.")
            return 0
        return 0