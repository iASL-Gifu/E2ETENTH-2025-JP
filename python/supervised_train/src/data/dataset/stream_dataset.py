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
            total_sequences_across_all_bags += len(bag_dataset) # SequenceStreamDatasetの__len__を呼び出す
        
        if self.batch_size == 0: # ゼロ除算回避
            # batch_sizeが0は想定外だが、安全のため総シーケンス数を返す
            return total_sequences_across_all_bags 

        # 全シーケンス数をバッチサイズで割って切り上げると、生成されるバッチの概算数になる
        # max(1, ...) は、データが少ない場合に少なくとも1を返すため
        return max(1, math.ceil(total_sequences_across_all_bags / self.batch_size))

        

    def _load_bag_datasets(self):
        """ルートディレクトリからすべてのSingleStreamBagIterableDatasetインスタンスを読み込む"""
        bag_datasets = []
        for bag_name in sorted(os.listdir(self.root_dir)):
            bag_dir = os.path.join(self.root_dir, bag_name)
            if not os.path.isdir(bag_dir):
                continue
            
            try:
                # SingleStreamBagIterableDatasetにはtransformはここでは渡さない（MultiStreamでまとめて行うため）
                # ただし、SingleStreamBagIterableDataset内でNumPyの型変換などを行う場合は渡す
                # 今回はSingleStreamBagIterableDatasetのtransformは基本的なものとし、
                # 最終的なTensor化はMultiStreamConcatIterableDatasetのtransformで行う想定
                single_dataset = SequenceStreamDataset(bag_dir, self.sequence_length, transform=None)
                # _is_valid_bagを確認して、有効なものだけを追加
                if single_dataset._is_valid_bag:
                    bag_datasets.append(single_dataset)
                else:
                    # _is_valid_bagがFalseの場合、すでにWarningが出ているので何もしない
                    pass
            except FileNotFoundError as e:
                print(f"Skipping {bag_dir} due to missing files: {e}")
            except ValueError as e:
                print(f"Skipping {bag_dir} due to data error: {e}")
        
        print(f"[INFO] Loaded {len(bag_datasets)} valid bags for multi-stream iteration.")
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
        worker_bag_indices = [] # 各ワーカーが現在担当している bag_datasets のインデックス
        
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
                # bagの数よりbatch_sizeが大きい場合、残りはNoneで埋める
                worker_iterators.append(None)
                worker_bag_indices.append(None)
        
        # 循環キューのカーソル（次に割り当てるシャッフル済みbagのインデックス）
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
                    # このストリームはすでに終了しているか、最初から割り当てられていない
                    # ダミーサンプルを追加
                    if not self.bag_datasets: # データがない場合はスキップ
                        continue
                    ref_scan_shape = (N,) + self.bag_datasets[0].scans.shape[1:]
                    ref_action_shape = (N,) + self.bag_datasets[0].actions.shape[1:]
                    
                    batch['scan_seq'].append(np.zeros(ref_scan_shape, dtype=np.float32))
                    batch['prev_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                    batch['target_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                    batch['is_first_seq'].append(True) # ダミーなので常にリセット
                    continue

                try:
                    # 単一ストリームから次のサンプルを取得
                    sample = next(worker_iterators[i])
                    num_active_streams += 1
                    
                    # データ拡張の適用
                    if self.augmentor and bag_idx is not None and bag_idx in episode_plans:
                        plan = episode_plans[bag_idx]
                        sample = self.augmentor.apply(sample, plan)
                    
                    # transformの適用（Tensor化など）
                    if self.transform:
                        sample = self.transform(sample)

                    # is_first_seqはSingleStreamBagIterableDatasetから取得したものを使用
                    is_first_seq = sample['is_first_seq'] # bool値

                    for key in ['scan_seq', 'prev_action_seq', 'target_action_seq']:
                        batch[key].append(sample[key])
                    batch['is_first_seq'].append(is_first_seq)

                except StopIteration:
                    # 現在のbagからのデータが尽きた場合
                    # 次のbagを割り当てるか、全bagを使い切っていたらストリームを終了させる
                    
                    # 次のシャッフルされたbagのインデックスを取得
                    if next_shuffled_bag_cursor < len(shuffled_bag_indices):
                        new_bag_idx = shuffled_bag_indices[next_shuffled_bag_cursor]
                        next_shuffled_bag_cursor += 1
                        
                        worker_bag_indices[i] = new_bag_idx
                        worker_iterators[i] = iter(self.bag_datasets[new_bag_idx])
                        
                        # 新しいbagの最初のサンプルを再試行して取得
                        try:
                            sample = next(worker_iterators[i])
                            num_active_streams += 1

                            # データ拡張の適用
                            if self.augmentor and new_bag_idx is not None and new_bag_idx in episode_plans:
                                plan = episode_plans[new_bag_idx]
                                sample = self.augmentor.apply(sample, plan)
                            
                            # transformの適用（Tensor化など）
                            if self.transform:
                                sample = self.transform(sample)

                            is_first_seq = sample['is_first_seq'] # bool値

                            for key in ['scan_seq', 'prev_action_seq', 'target_action_seq']:
                                batch[key].append(sample[key])
                            batch['is_first_seq'].append(is_first_seq)

                        except StopIteration:
                            # 割り当てられた新しいbagも空（非常に短い場合など）
                            # このストリームは終了
                            worker_iterators[i] = None
                            worker_bag_indices[i] = None
                            # ダミーサンプルを追加
                            if not self.bag_datasets: continue
                            ref_scan_shape = (N,) + self.bag_datasets[0].scans.shape[1:]
                            ref_action_shape = (N,) + self.bag_datasets[0].actions.shape[1:]
                            batch['scan_seq'].append(np.zeros(ref_scan_shape, dtype=np.float32))
                            batch['prev_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                            batch['target_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                            batch['is_first_seq'].append(True)
                    else:
                        # 全てのbagが処理され、新しいbagを割り当てられない
                        # このストリームは終了
                        worker_iterators[i] = None
                        worker_bag_indices[i] = None
                        # ダミーサンプルを追加
                        if not self.bag_datasets: continue
                        ref_scan_shape = (N,) + self.bag_datasets[0].scans.shape[1:]
                        ref_action_shape = (N,) + self.bag_datasets[0].actions.shape[1:]
                        batch['scan_seq'].append(np.zeros(ref_scan_shape, dtype=np.float32))
                        batch['prev_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                        batch['target_action_seq'].append(np.zeros(ref_action_shape, dtype=np.float32))
                        batch['is_first_seq'].append(True)

            # アクティブなストリームが一つもなければループを終了
            if num_active_streams == 0:
                break

            # バッチ内のデータをPyTorch Tensorに変換してyield
            final_batch = {}
            for key, val in batch.items():
                if key == 'is_first_seq':
                    # is_first_seq は bool なので、torch.bool 型でテンソル化
                    final_batch[key] = torch.tensor(val, dtype=torch.bool)
                elif isinstance(val, list) and val and isinstance(val[0], torch.Tensor):
                    # val が既に Tensor のリストの場合 (transformによってTensor化されている場合)
                    final_batch[key] = torch.stack(val)
                elif isinstance(val, list) and val and isinstance(val[0], np.ndarray):
                    # val が NumPy array のリストの場合 (transformがNoneまたはNumPyを返す場合)
                    # ★★★ ここで NumPy 配列のリストを単一の NumPy 配列に結合し、float32にキャスト
                    combined_np_array = np.array(val, dtype=np.float32)
                    # その後、結合された NumPy 配列から Tensor を作成
                    final_batch[key] = torch.from_numpy(combined_np_array)
                else:
                    # その他のケース（例えば、NumPy配列ではないがテンソルにしたい数値リストなど）
                    # 万が一に備えて dtype を指定
                    final_batch[key] = torch.tensor(val, dtype=torch.float32)
            # is_first_seqはすでにTensorになっているのでそのまま
            if isinstance(final_batch['is_first_seq'], list): # Transformによってはまだリストの場合がある
                final_batch['is_first_seq'] = torch.tensor(final_batch['is_first_seq'], dtype=torch.bool)


            yield final_batch
