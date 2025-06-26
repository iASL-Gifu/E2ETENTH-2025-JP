import torch
import os
import random
import numpy as np
import pygame
import time
import hydra
import torch
import matplotlib.pyplot as plt
from collections import deque

class LiDARCurvatureCNN(torch.nn.Module):
    """LiDAR曲率分類用の1D CNNモデル（推論用）"""
    def __init__(self, num_classes, input_dim=2, dropout_rate=0.3):
        super(LiDARCurvatureCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # 第1ブロック: 局所的な特徴抽出
        self.conv1_1 = torch.nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv1_2 = torch.nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.pool1 = torch.nn.MaxPool1d(2)
        
        # 第2ブロック: 中程度の範囲の特徴抽出
        self.conv2_1 = torch.nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv2_2 = torch.nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.pool2 = torch.nn.MaxPool1d(2)
        
        # 第3ブロック: より広い範囲の特徴抽出
        self.conv3_1 = torch.nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.conv3_2 = torch.nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.pool3 = torch.nn.MaxPool1d(2)
        
        # 第4ブロック: 大域的な特徴抽出
        self.conv4_1 = torch.nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv4_2 = torch.nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn4 = torch.nn.BatchNorm1d(256)
        
        # グローバル平均プーリング
        self.global_avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        
        # 分類ヘッド
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, num_classes)
        )
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 入力: (batch_size, 100, 2) -> (batch_size, 2, 100)
        x = x.transpose(1, 2)
        
        # 第1ブロック
        x = torch.nn.functional.relu(self.conv1_1(x))
        x = torch.nn.functional.relu(self.bn1(self.conv1_2(x)))
        x = self.pool1(x)
        
        # 第2ブロック
        x = torch.nn.functional.relu(self.conv2_1(x))
        x = torch.nn.functional.relu(self.bn2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # 第3ブロック
        x = torch.nn.functional.relu(self.conv3_1(x))
        x = torch.nn.functional.relu(self.bn3(self.conv3_2(x)))
        x = self.pool3(x)
        
        # 第4ブロック
        x = torch.nn.functional.relu(self.conv4_1(x))
        x = torch.nn.functional.relu(self.bn4(self.conv4_2(x)))
        
        # グローバル特徴抽出
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        
        # 分類
        x = self.classifier(x)
        
        return x
    
    
    
class F1TenthLiDARInferenceEngine:
    """F1Tenth環境用のLiDAR曲率分類推論エンジン"""
    
    def __init__(self, model_path=None, device='cuda'):
        """
        推論エンジンの初期化
        
        Args:
            model_path: 学習済みモデルのパス（Noneの場合は最新モデルを自動検索）
            device: 使用するデバイス
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # モデルパスの自動検索
        if model_path is None:
            model_path = self._find_latest_model()
        
        self.model_path = model_path
        
        # モデルと情報を読み込み
        self.model, self.model_info = self._load_model()
        
        # ラベルマッピングを取得
        self.label_mapping = self.model_info.get('label_mapping', {})
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # 統計情報の初期化
        self.prediction_history = deque(maxlen=100)  # 最新100予測を保持
        self.class_counts = {i: 0 for i in range(self.model.num_classes)}
        self.total_predictions = 0
        
        print(f"🎯 F1Tenth LiDAR Inference Engine Initialized!")
        print(f"   📱 Device: {self.device}")
        print(f"   📄 Model: {os.path.basename(self.model_path)}")
        print(f"   🎯 Classes: {self.model.num_classes}")
        print(f"   🏷️ Label mapping: {self.label_mapping}")
    
    def _find_latest_model(self, model_dir='./models'):
        """最新のモデルファイルを検索"""
        import glob
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        model_files = glob.glob(os.path.join(model_dir, "lidar_curvature_model_*.pth"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # 最新のファイルを返す
        latest_model = sorted(model_files, reverse=True)[0]
        print(f"📁 Auto-selected model: {os.path.basename(latest_model)}")
        return latest_model
    
    def _load_model(self):
        """モデルを読み込む"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # モデル情報を読み込み（PyTorch 2.6対応）
        try:
            # 安全なグローバル設定を追加してからロード
            safe_globals = [
                'numpy._core.multiarray.scalar',
                'numpy.core.multiarray.scalar',
                'numpy.dtype',
                'numpy.ndarray',
                'builtins.dict',
                'builtins.list',
                'builtins.tuple',
                'builtins.int',
                'builtins.float',
                'builtins.str',
                'builtins.bool'
            ]
            
            # 安全なグローバル設定のコンテキストマネージャーを使用
            with torch.serialization.safe_globals(safe_globals):
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                print(f"   ✅ Model loaded with secure settings")
            
        except Exception as e1:
            try:
                # セキュリティ設定が厳しい場合は、信頼できるファイルとして読み込み
                print(f"   🔒 Using fallback loading method for trusted model file...")
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                print(f"   ✅ Model loaded with fallback method")
                
            except Exception as e2:
                raise RuntimeError(f"Failed to load model with both secure and fallback methods.\n"
                                 f"Secure error: {e1}\n"
                                 f"Fallback error: {e2}")
        
        # モデルを作成
        num_classes = checkpoint['num_classes']
        model_config = checkpoint.get('model_config', {})
        
        model = LiDARCurvatureCNN(
            num_classes=num_classes,
            input_dim=model_config.get('input_dim', 2),
            dropout_rate=model_config.get('dropout_rate', 0.3)
        )
        
        # 重みを読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, checkpoint
    
    def convert_f1tenth_scan_to_xy(self, scan_data, angle_min=-2.35, angle_max=2.35):
        """
        F1TenthのLiDARスキャンデータ（距離）をXY座標に変換
        
        Args:
            scan_data: F1TenthのLiDARスキャンデータ（距離の配列）
            angle_min: 最小角度（ラジアン）
            angle_max: 最大角度（ラジアン）
        
        Returns:
            xy_data: (N, 2) のXY座標データ
        """
        # 角度の生成
        num_points = len(scan_data)
        angles = np.linspace(angle_min, angle_max, num_points)
        
        # 無効な値（inf, nan）を処理
        valid_mask = np.isfinite(scan_data) & (scan_data > 0)
        scan_data = np.where(valid_mask, scan_data, 10.0)  # 無効値は10mに設定
        
        # 極座標からXY座標に変換
        x = scan_data * np.cos(angles)
        y = scan_data * np.sin(angles)
        
        # XY座標を結合
        xy_data = np.column_stack([x, y])
        
        return xy_data
    
    def preprocess_f1tenth_lidar(self, scan_data):
        """
        F1TenthのLiDARデータを推論用に前処理
        
        Args:
            scan_data: F1TenthのLiDARスキャンデータ
        
        Returns:
            processed_data: 推論用Tensor (1, 100, 2)
        """
        # スキャンデータをXY座標に変換
        xy_data = self.convert_f1tenth_scan_to_xy(scan_data)
        
        # 100点になるようにリサンプリング
        if len(xy_data) != 100:
            # 線形補間でリサンプリング
            indices = np.linspace(0, len(xy_data) - 1, 100)
            x_interp = np.interp(indices, np.arange(len(xy_data)), xy_data[:, 0])
            y_interp = np.interp(indices, np.arange(len(xy_data)), xy_data[:, 1])
            xy_data = np.column_stack([x_interp, y_interp])
        
        # バッチ次元を追加して Tensor に変換
        xy_data = xy_data.reshape(1, 100, 2)
        return torch.FloatTensor(xy_data).to(self.device)
    
    def predict_realtime(self, scan_data, return_probabilities=True):
        """
        リアルタイム推論
        
        Args:
            scan_data: F1TenthのLiDARスキャンデータ
            return_probabilities: 確率を返すかどうか
        
        Returns:
            result: 推論結果の辞書
        """
        # データの前処理
        processed_data = self.preprocess_f1tenth_lidar(scan_data)
        
        # 推論実行
        with torch.no_grad():
            outputs = self.model(processed_data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).cpu().item()
            confidence = probabilities[0, predicted_class].cpu().item()
            
            # 元のラベルに変換
            original_label = self.reverse_label_mapping.get(predicted_class, predicted_class)
        
        # 統計情報を更新
        self.prediction_history.append({
            'class': predicted_class,
            'confidence': confidence,
            'timestamp': time.time()
        })
        self.class_counts[predicted_class] += 1
        self.total_predictions += 1
        
        # 結果をまとめる
        result = {
            'predicted_class': predicted_class,
            'original_label': original_label,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy() if return_probabilities else None,
            'raw_outputs': outputs[0].cpu().numpy()
        }
        
        return result
    
    def get_statistics(self):
        """統計情報を取得"""
        if self.total_predictions == 0:
            return {}
        
        # 最近の予測の平均信頼度
        recent_confidences = [p['confidence'] for p in self.prediction_history]
        avg_confidence = np.mean(recent_confidences) if recent_confidences else 0
        
        # クラス分布
        class_distribution = {}
        for class_idx, count in self.class_counts.items():
            percentage = count / self.total_predictions * 100
            class_distribution[class_idx] = {
                'count': count,
                'percentage': percentage
            }
        
        return {
            'total_predictions': self.total_predictions,
            'average_confidence': avg_confidence,
            'class_distribution': class_distribution,
            'recent_predictions': len(self.prediction_history)
        }
    
    def visualize_realtime_stats(self):
        """リアルタイム統計の可視化"""
        stats = self.get_statistics()
        
        if stats['total_predictions'] == 0:
            return
        
        # クラス分布の表示
        print(f"\n📊 Real-time Statistics (Total: {stats['total_predictions']})")
        print(f"   Average Confidence: {stats['average_confidence']:.3f}")
        print(f"   Class Distribution:")
        
        for class_idx, dist in stats['class_distribution'].items():
            orig_label = self.reverse_label_mapping.get(class_idx, class_idx)
            print(f"     Class {class_idx} (orig: {orig_label}): {dist['count']} ({dist['percentage']:.1f}%)")


class StabilizedLiDARInference:
    """
    5ステップ連続で同じクラスが予測された場合のみ更新するLiDAR推論の安定化クラス

    """


    def convert_scan(self, scans, scan_range: float=30.0):
        scans = scans / scan_range
        scans = np.clip(scans, 0, 1)
        return scans    
    def __init__(self, inference_engine, stability_window=5):
        """
        Args:
            inference_engine: F1TenthLiDARInferenceEngine インスタンス
            stability_window: 安定化のためのウィンドウサイズ（デフォルト5ステップ）
        """
        self.inference_engine = inference_engine
        self.stability_window = stability_window
        
        # 過去の予測結果を保持
        self.prediction_history = deque(maxlen=stability_window)
        
        # 現在の安定したクラス
        self.stable_class = None
        self.stable_confidence = 0.0
        self.stable_result = None
        
        # 統計情報
        self.total_predictions = 0
        self.stable_updates = 0
        self.last_update_step = 0
        
        print(f"🎯 Stabilized LiDAR Inference initialized with {stability_window}-step window")
    
    def predict_and_stabilize(self, scan_data, current_step=None):
        """
        LiDAR推論を実行し、安定性をチェックして必要に応じて更新
        
        Args:
            scan_data: LiDARスキャンデータ
            current_step: 現在のステップ数（統計用、オプション）
        
        Returns:
            dict: {
                'inference_result': 最新の推論結果,
                'stable_result': 安定した推論結果（更新された場合）,
                'is_updated': 安定した結果が更新されたかどうか,
                'stability_info': 安定性に関する情報
            }
        """
        # 最新の推論を実行
        
        scan_data=self.convert_scan(scan_data)
        inference_result = self.inference_engine.predict_realtime(scan_data)
        predicted_class = inference_result['predicted_class']
        confidence = inference_result['confidence']
        
        # 予測履歴に追加
        self.prediction_history.append({
            'class': predicted_class,
            'confidence': confidence,
            'step': current_step
        })
        
        self.total_predictions += 1
        
        # 安定性をチェック
        is_stable, stability_info = self._check_stability()
        is_updated = False
        
        if is_stable:
            # 安定したクラスが変更された場合のみ更新
            if self.stable_class != predicted_class:
                self.stable_class = predicted_class
                self.stable_confidence = confidence
                self.stable_result = inference_result.copy()
                self.stable_updates += 1
                self.last_update_step = current_step if current_step else self.total_predictions
                is_updated = True
                
        
        return {
            'inference_result': inference_result,
            'stable_result': self.stable_result,
            'is_updated': is_updated,
            'stability_info': stability_info
        }
    
    def _check_stability(self):
        """
        予測の安定性をチェック
        
        Returns:
            tuple: (is_stable: bool, stability_info: dict)
        """
        if len(self.prediction_history) < self.stability_window:
            return False, {
                'status': 'insufficient_data',
                'window_filled': len(self.prediction_history),
                'required': self.stability_window
            }
        
        # 最新のN個の予測が全て同じクラスかチェック
        recent_classes = [pred['class'] for pred in self.prediction_history]
        unique_classes = set(recent_classes)
        
        if len(unique_classes) == 1:
            # 全て同じクラス
            stable_class = recent_classes[0]
            avg_confidence = np.mean([pred['confidence'] for pred in self.prediction_history])
            
            return True, {
                'status': 'stable',
                'stable_class': stable_class,
                'window_size': len(recent_classes),
                'average_confidence': avg_confidence,
                'confidence_std': np.std([pred['confidence'] for pred in self.prediction_history])
            }
        else:
            # まだ不安定
            return False, {
                'status': 'unstable',
                'unique_classes': list(unique_classes),
                'class_counts': {cls: recent_classes.count(cls) for cls in unique_classes},
                'window_size': len(recent_classes)
            }
    
    def get_current_stable_class(self):
        """現在の安定したクラスを取得"""
        return {
            'stable_class': self.stable_class,
            'stable_confidence': self.stable_confidence,
            'stable_result': self.stable_result
        }
    
    def get_statistics(self):
        """統計情報を取得"""
        stability_rate = self.stable_updates / max(self.total_predictions, 1) * 100
        
        return {
            'total_predictions': self.total_predictions,
            'stable_updates': self.stable_updates,
            'stability_rate': stability_rate,
            'last_update_step': self.last_update_step,
            'current_stable_class': self.stable_class,
            'window_size': self.stability_window
        }
    
    def visualize_stability_stats(self):
        """安定性統計の可視化"""
        stats = self.get_statistics()
        
        print(f"\n📊 Stability Statistics:")
        print(f"   Total Predictions: {stats['total_predictions']}")
        print(f"   Stable Updates: {stats['stable_updates']}")
        print(f"   Stability Rate: {stats['stability_rate']:.1f}%")
        print(f"   Current Stable Class: {stats['current_stable_class']}")
        print(f"   Last Update Step: {stats['last_update_step']}")
        print(f"   Window Size: {stats['window_size']}")