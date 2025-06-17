import os
import numpy as np
import torch
from torch.utils.data import IterableDataset
import random
import math

from .transform import StreamAugmentor 

class SequenceStreamDataset(IterableDataset):
    """
    単一のルートディレクトリ (bag) から、オーバーラップするシーケンスを逐次的に生成するデータセット。
    データ拡張は行わず、transformによる基本的なデータ整形のみを行う。
    """
    def __init__(self, bag_dir, sequence_length=10, transform=None):
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
        
        self.bag_dir = bag_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # データのロード
        scans_path = os.path.join(bag_dir, 'scans.npy')
        steers_path = os.path.join(bag_dir, 'steers.npy')
        speeds_path = os.path.join(bag_dir, 'speeds.npy')
        
        if not all(os.path.exists(p) for p in [scans_path, steers_path, speeds_path]):
            raise FileNotFoundError(f"Missing one or more .npy files in {bag_dir}")
        
        self.scans = np.load(scans_path)
        steers = np.load(steers_path)
        speeds = np.load(speeds_path)
        self.actions = np.stack([steers, speeds], axis=1).astype(np.float32)
        
        if len(self.scans) < self.sequence_length + 1:
            # シーケンス長+1 に満たない場合は、このバッグからはサンプルを生成しない
            # __iter__で空のイテレータを返すことで対応
            self._is_valid_bag = False
            print(f"Warning: Bag {bag_dir} too short for sequence_length {sequence_length}. Skipping.")
        else:
            self._is_valid_bag = True

    def __len__(self):
        """
        このバッグから生成可能なシーケンスの総数を概算して返します。
        オーバーラップするシーケンス数を考慮します。
        """
        if not self._is_valid_bag:
            return 0 # 無効なバッグは0を返す

        num_frames = len(self.actions)
        N = self.sequence_length

        # 例えば、100フレームでシーケンス長10の場合、0-9, 1-10, ..., 90-99 の91シーケンス
        total_overlapping_sequences = max(0, num_frames - N + 1)
        
        if total_overlapping_sequences == 0:
            return 0
        else:
            return max(1, math.ceil(total_overlapping_sequences / N))


    def __iter__(self):
        if not self._is_valid_bag:
            return iter([]) # 有効でないバッグは空のイテレータを返す

        N = self.sequence_length
        num_frames = len(self.actions)

        # オーバーラップしながらシーケンスを生成
        for start_frame in range(num_frames - N + 1):
            current_scan_seq = self.scans[start_frame : start_frame + N]
            current_target_action_seq = self.actions[start_frame : start_frame + N]

            # prev_action_seq の計算
            if start_frame == 0:
                # 最初のシーケンスの場合、prev_actionの最初の要素はダミー（ゼロ）
                dummy_prev_action = np.zeros_like(self.actions[0])
                current_prev_action_seq = np.concatenate(
                    (dummy_prev_action[np.newaxis, :], self.actions[start_frame : start_frame + N - 1]),
                    axis=0
                )
            else:
                # 通常のシーケンスの場合、1つ前のフレームからN個のprev_action
                current_prev_action_seq = self.actions[start_frame - 1 : start_frame + N - 1]

            sample = {
                'scan_seq': current_scan_seq.copy(), # augmentorが変更できるようにコピーを渡す
                'prev_action_seq': current_prev_action_seq.copy(),
                'target_action_seq': current_target_action_seq.copy(),
                'is_first_seq': start_frame == 0 # このシーケンスがバッグの最初かどうか
            }

            if self.transform:
                sample = self.transform(sample) # ここではTensor化などはまだ行わない方が良い場合も

            # SingleStreamBagIterableDatasetはNumPy配列を返す
            yield sample
            

class ConcatStreamDataset(IterableDataset):
    """
    複数のSingleStreamBagIterableDatasetインスタンスを結合し、
    指定されたbatch_sizeで並行してシーケンスを供給するデータセット。
    StreamAugmentorによるエピソード単位のデータ拡張をここで適用する。
    """
    def __init__(self, root_dir, batch_size, sequence_length, 
                 augmentor: StreamAugmentor = None, transform=None):
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
        self.augmentor = augmentor
        self.transform = transform 

        self.bag_datasets = self._load_bag_datasets()

        if not self.bag_datasets:
            raise ValueError("No valid bag datasets were found in the root directory.")
        
    def __len__(self):
        """
        このデータセットが生成するバッチの総数を概算して返します。
        内部の各SequenceStreamDatasetの長さを合計し、batch_sizeで割って切り上げます。
        """
        total_sequences_across_all_bags = 0
        
        # 各bag_dataset（SequenceStreamDatasetインスタンス）の長さを合計
        for bag_dataset in self.bag_datasets:
            total_sequences_across_all_bags += len(bag_dataset)
        
        if self.batch_size == 0:
            return total_sequences_across_all_bags 

        return max(1, math.ceil(total_sequences_across_all_bags / self.batch_size))

    def _load_bag_datasets(self):
        """ルートディレクトリを再帰的に探索し、すべての有効なデータセットインスタンスを読み込む"""
        bag_datasets = []
        print(f"[INFO] Recursively searching for bag data in '{self.root_dir}'...")

        # os.walkを使ってディレクトリを再帰的に探索
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            try:
                # このディレクトリでデータセットの作成を試みる
                # SequenceStreamDatasetのコンストラクタ内で、必要なファイルが存在するか等の検証が行われる想定
                single_dataset = SequenceStreamDataset(dirpath, self.sequence_length, transform=None)
                
                # _is_valid_bagプロパティで、有効なデータセットか確認
                if single_dataset._is_valid_bag:
                    print(f"[INFO] Found valid bag data at: {dirpath}")
                    bag_datasets.append(single_dataset)
                # else の場合は、SequenceStreamDataset内で警告が出力されるか、
                # または単に無効なデータセットとして扱われる

            except (FileNotFoundError, ValueError) as e:
                # FileNotFoundError: 必須ファイルがない場合
                # ValueError: データに問題がある場合（例：データが短すぎるなど）
                # これらのエラーは、現在のディレクトリが有効なデータセットではないことを示すため、
                # スキップして次の探索を続ける
                # print(f"Skipping {dirpath} due to an error: {e}") # 詳細なログが必要な場合は有効化
                pass
        
        print("-" * 50)
        if not bag_datasets:
            print("[INFO] No valid bag datasets were found.")
        else:
            # 読み込み順序の再現性を確保するために、ディレクトリパスでソート
            bag_datasets.sort(key=lambda ds: ds.bag_dir)
            print(f"[INFO] Loaded {len(bag_datasets)} valid bags for multi-stream iteration.")
        print("-" * 50)

        return bag_datasets

    def __iter__(self):
        """データセットのイテレータを返す。"""
        
        # 各バッグ（エピソード）に対するデータ拡張計画を事前に立てる
        episode_plans = {}
        if self.augmentor:
            for i, bag_dataset in enumerate(self.bag_datasets):
                episode_plans[i] = self.augmentor.plan_for_episode()

        # 各ワーカー（ストリーム）のイテレータと、現在担当しているbagのインデックスを保持
        worker_iterators = []
        worker_bag_indices = []
        
        # 全てのbagのインデックスをシャッフルし、各ワーカーに割り当てる
        shuffled_bag_indices = list(range(len(self.bag_datasets)))
        random.shuffle(shuffled_bag_indices)

        # 各ワーカーに最初のbagを割り当てる
        for i in range(self.batch_size):
            if i < len(shuffled_bag_indices):
                bag_idx = shuffled_bag_indices[i]
                worker_iterators.append(iter(self.bag_datasets[bag_idx]))
                worker_bag_indices.append(bag_idx)
            else:
                worker_iterators.append(None)
                worker_bag_indices.append(None)
        
        next_shuffled_bag_cursor = self.batch_size
        N = self.sequence_length

        while True:
            batch = {
                'scan_seq': [], 'prev_action_seq': [], 'target_action_seq': [], 'is_first_seq': []
            }
            num_active_streams = 0

            for i in range(self.batch_size):
                sample = None
                bag_idx = worker_bag_indices[i]

                if worker_iterators[i] is None:
                    if not self.bag_datasets: continue
                    ref_scan_shape = (N,) + self.bag_datasets[0].scans.shape[1:]
                    ref_action_shape = (N,) + self.bag_datasets[0].actions.shape[1:]
                    
                    batch['scan_seq'].append(np.zeros(ref_scan_shape, dtype=np.float32))
                    batch['prev_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                    batch['target_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                    batch['is_first_seq'].append(True)
                    continue

                try:
                    sample = next(worker_iterators[i])
                    num_active_streams += 1
                    
                    if self.augmentor and bag_idx is not None and bag_idx in episode_plans:
                        sample = self.augmentor.apply(sample, episode_plans[bag_idx])
                    
                    if self.transform:
                        sample = self.transform(sample)

                    for key in ['scan_seq', 'prev_action_seq', 'target_action_seq']:
                        batch[key].append(sample[key])
                    batch['is_first_seq'].append(sample['is_first_seq'])

                except StopIteration:
                    if next_shuffled_bag_cursor < len(shuffled_bag_indices):
                        new_bag_idx = shuffled_bag_indices[next_shuffled_bag_cursor]
                        next_shuffled_bag_cursor += 1
                        
                        worker_bag_indices[i] = new_bag_idx
                        worker_iterators[i] = iter(self.bag_datasets[new_bag_idx])
                        
                        try:
                            sample = next(worker_iterators[i])
                            num_active_streams += 1

                            if self.augmentor and new_bag_idx is not None and new_bag_idx in episode_plans:
                                sample = self.augmentor.apply(sample, episode_plans[new_bag_idx])
                            
                            if self.transform:
                                sample = self.transform(sample)

                            for key in ['scan_seq', 'prev_action_seq', 'target_action_seq']:
                                batch[key].append(sample[key])
                            batch['is_first_seq'].append(sample['is_first_seq'])

                        except StopIteration:
                            worker_iterators[i] = None
                            worker_bag_indices[i] = None
                            if not self.bag_datasets: continue
                            ref_scan_shape = (N,) + self.bag_datasets[0].scans.shape[1:]
                            ref_action_shape = (N,) + self.bag_datasets[0].actions.shape[1:]
                            batch['scan_seq'].append(np.zeros(ref_scan_shape, dtype=np.float32))
                            batch['prev_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                            batch['target_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                            batch['is_first_seq'].append(True)
                    else:
                        worker_iterators[i] = None
                        worker_bag_indices[i] = None
                        if not self.bag_datasets: continue
                        ref_scan_shape = (N,) + self.bag_datasets[0].scans.shape[1:]
                        ref_action_shape = (N,) + self.bag_datasets[0].actions.shape[1:]
                        batch['scan_seq'].append(np.zeros(ref_scan_shape, dtype=np.float32))
                        batch['prev_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                        batch['target_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                        batch['is_first_seq'].append(True)

            if num_active_streams == 0:
                break

            final_batch = {}
            for key, val in batch.items():
                if not val: continue
                if isinstance(val[0], torch.Tensor):
                    final_batch[key] = torch.stack(val)
                elif isinstance(val[0], np.ndarray):
                    final_batch[key] = torch.from_numpy(np.array(val, dtype=np.float32))
                elif key == 'is_first_seq':
                     final_batch[key] = torch.tensor(val, dtype=torch.bool)
                else:
                    final_batch[key] = torch.tensor(val, dtype=torch.float32)

            yield final_batch

