import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from rcl_interfaces.msg import SetParametersResult
import torch
import numpy as np
import os

from collections import deque

from .models.sac import Actor  # SACモデル定義
from .models.LiDAR_CNN import F1TenthLiDARInferenceEngine, StabilizedLiDARInference

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        # デバイス設定（CUDA 優先）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[DEVICE] Using device: {self.device}")

        # パラメータ宣言
        self.declare_parameter('ckpt_path', 'checkpoint.pt')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('downsample_num', 100)
        self.declare_parameter('lidar_dim', 1080)
        self.declare_parameter('action_dim', 2)
        self.declare_parameter('hidden_dim', 256)
        self.declare_parameter('output_steering_gain', 1.0)
        self.declare_parameter('output_throttle_gain', 1.0)
        self.declare_parameter('use_adaptive_downsampling', True)
        self.declare_parameter("curvature_inference", False)  # カーブ推論の有効/無効
        self.declare_parameter('scan_window_size', 1)
        self.declare_parameter('stability_window', 5)  # 安定化ウィンドウサイズ
        self.declare_parameter('enable_speed_adjustment', False)  # 速度調整の有効/無効
        self.declare_parameter('inference_display_interval', 50)  # 推論結果表示間隔
        self.declare_parameter('final_gain', 1.2) 
        # パラメータの読み込み
        self.load_parameters()

        # スキャンデータの履歴を保持するためのウィンドウ
        self.scan_window = deque(maxlen=self.scan_window_size)

        # SACモデルをデバイス上に配置
        self.model = self.load_model(self.ckpt_path,
                                     self.lidar_dim,
                                     self.action_dim,
                                     self.hidden_dim)

        # LiDAR推論エンジンの初期化
        self.lidar_inference = None
        self.stabilized_inference = None
        self.inference_enabled = False
        self.modelpath="/home/tamiya/E2ETENTH-2025-JP/ckpts_kei/curvarate/lidar_curvature_model_weighted_20250623_135856.pth"
        if self.curvature_inference:
            try:
                self.lidar_inference = F1TenthLiDARInferenceEngine(model_path=self.modelpath,device=self.device)
                self.stabilized_inference = StabilizedLiDARInference(
                    inference_engine=self.lidar_inference,
                    stability_window=self.stability_window
                )
                self.inference_enabled = True
                self.get_logger().info("✅ LiDAR curvature inference engine loaded successfully!")
                self.get_logger().info("✅ Stabilized inference wrapper initialized!")
            except Exception as e:
                self.get_logger().error(f"⚠️ Could not load LiDAR inference engine: {e}")
                self.get_logger().info("   Continuing without LiDAR curvature inference...")
                self.inference_enabled = False

        # 統計用カウンタ
        self.step_count = 0

        # 動的パラメータ変更時のコールバック登録
        self.add_on_set_parameters_callback(self.on_param_change)

        # ROS I/O
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.publisher = self.create_publisher(
            AckermannDrive,
            '/cmd_drive',
            10)

        self.get_logger().info(
            f"[STARTED] SAC model: {self.ckpt_path}, downsample_num: {self.downsample_num}, "
            f"max_range: {self.max_range}, adaptive: {self.use_adaptive_downsampling}, "
            f"curvature_inference: {self.inference_enabled}"
        )

    def load_parameters(self):
        self.ckpt_path = self.get_parameter('ckpt_path').get_parameter_value().string_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.downsample_num = self.get_parameter('downsample_num').get_parameter_value().integer_value
        self.lidar_dim = self.get_parameter('lidar_dim').get_parameter_value().integer_value
        self.action_dim = self.get_parameter('action_dim').get_parameter_value().integer_value
        self.hidden_dim = self.get_parameter('hidden_dim').get_parameter_value().integer_value
        self.output_steering_gain = self.get_parameter('output_steering_gain').get_parameter_value().double_value
        self.output_throttle_gain = self.get_parameter('output_throttle_gain').get_parameter_value().double_value
        self.use_adaptive_downsampling = self.get_parameter('use_adaptive_downsampling').get_parameter_value().bool_value
        self.curvature_inference = self.get_parameter("curvature_inference").get_parameter_value().bool_value
        self.scan_window_size = self.get_parameter('scan_window_size').get_parameter_value().integer_value
        self.stability_window = self.get_parameter('stability_window').get_parameter_value().integer_value
        self.enable_speed_adjustment = self.get_parameter('enable_speed_adjustment').get_parameter_value().bool_value
        self.inference_display_interval = self.get_parameter('inference_display_interval').get_parameter_value().integer_value
        self.final_gain = self.get_parameter('final_gain').get_parameter_value().double_value
    def on_param_change(self, params):
        success = True
        reason = ""

        for param in params:
            if param.name == 'ckpt_path':
                try:
                    self.model = self.load_model(param.value,
                                                 self.lidar_dim,
                                                 self.action_dim,
                                                 self.hidden_dim)
                    self.ckpt_path = param.value
                    self.get_logger().info(f"[RELOADED] Model from {param.value}")
                except Exception as e:
                    success = False
                    reason += f" Model reload failed: {e}"
            elif param.name == 'max_range':
                self.max_range = param.value
                self.get_logger().info(f"[UPDATED] max_range: {self.max_range}")
            elif param.name == 'downsample_num':
                self.downsample_num = param.value
                self.get_logger().info(f"[UPDATED] downsample_num: {self.downsample_num}")
            elif param.name == 'lidar_dim':
                self.lidar_dim = param.value
            elif param.name == 'action_dim':
                self.action_dim = param.value
            elif param.name == 'hidden_dim':
                self.hidden_dim = param.value
            elif param.name == 'output_steering_gain':
                self.output_steering_gain = param.value
                self.get_logger().info(f"[UPDATED] output_steering_gain: {self.output_steering_gain}")
            elif param.name == 'output_throttle_gain':
                self.output_throttle_gain = param.value
                self.get_logger().info(f"[UPDATED] output_throttle_gain: {self.output_throttle_gain}")
            elif param.name == 'use_adaptive_downsampling':
                self.use_adaptive_downsampling = param.value
                self.get_logger().info(f"[UPDATED] use_adaptive_downsampling: {self.use_adaptive_downsampling}")
            elif param.name == 'curvature_inference':
                self.curvature_inference = param.value
                self.get_logger().info(f"[UPDATED] curvature_inference: {self.curvature_inference}")
            elif param.name == 'scan_window_size':
                self.scan_window_size = param.value
                self.scan_window = deque(maxlen=self.scan_window_size)
                self.get_logger().info(f"[UPDATED] scan_window_size: {self.scan_window_size}")
            elif param.name == 'stability_window':
                self.stability_window = param.value
                self.get_logger().info(f"[UPDATED] stability_window: {self.stability_window}")
            elif param.name == 'enable_speed_adjustment':
                self.enable_speed_adjustment = param.value
                self.get_logger().info(f"[UPDATED] enable_speed_adjustment: {self.enable_speed_adjustment}")
            elif param.name == 'inference_display_interval':
                self.inference_display_interval = param.value
                self.get_logger().info(f"[UPDATED] inference_display_interval: {self.inference_display_interval}")

        # lidar_dim, action_dim, hidden_dim はモデル再読み込み時に使われるため、
        # ログ出力のみにとどめ、値の更新のみ行う
        if any(p.name in ['lidar_dim', 'action_dim', 'hidden_dim'] for p in params):
            self.get_logger().info("Model architecture parameters updated. Reload model (change ckpt_path) to apply.")

        return SetParametersResult(successful=success, reason=reason)

    def load_model(self, path, lidar_dim, action_dim, hidden_dim):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model = Actor(lidar_dim=self.downsample_num,
                      action_dim=action_dim,
                      hidden_dim=hidden_dim)
        checkpoint = torch.load(path, map_location=self.device)
            
        if 'actor' in checkpoint:
            # SACAgentから保存された場合
            model.load_state_dict(checkpoint['actor'])
            self.get_logger().info(f"[✔] SAC actor weights successfully loaded from {path}")
        elif 'actor_state_dict' in checkpoint:
            # SACAgentのsave_modelから保存された場合
            model.load_state_dict(checkpoint['actor_state_dict'])
            self.get_logger().info(f"[✔] SAC actor state dict successfully loaded from {path}")
        else:
            # 直接Actorの状態辞書が保存されている場合
            model.load_state_dict(checkpoint)
            self.get_logger().info(f"[✔] Actor weights successfully loaded from {path}")
            
        model.eval()
        model.to(self.device)
        return model

    def downsample_single_frame(self, scan_data: np.ndarray, target_size: int) -> np.ndarray:
        """
        1つの2D LiDARフレーム（1080ビーム、-135°～+135°）を適応的ダウンサンプリング
        
        Args:
            scan_data: LiDARスキャンデータ（1D配列、通常1080要素）
            target_size: 目標サンプル数
        
        Returns:
            ダウンサンプリングされたデータ
        """
        if target_size is None or scan_data.size == target_size:
            return scan_data
        
        # 2D LiDARスキャンの角度配列を生成（-135°から+135°の範囲）
        start_angle = -135 * np.pi / 180  # -135度をラジアンに変換
        end_angle = 135 * np.pi / 180     # +135度をラジアンに変換
        angles = np.linspace(start_angle, end_angle, scan_data.size)
        
        # 前方セクターの定義（前方向から±30度）
        front_angle_range = np.pi / 6  # 30度をラジアンで
        is_front = np.abs(angles) <= front_angle_range
        
        # サンプリング比率の計算
        front_ratio = 0.7  # 前方エリアに70%のサンプル
        
        front_count = int(target_size * front_ratio)
        side_count = target_size - front_count
        
        # 前方エリアと側面エリアのインデックスを取得
        front_indices = np.where(is_front)[0]
        side_indices = np.where(~is_front)[0]
        
        selected_indices = []
        
        # 前方エリアを高密度でサンプリング
        if len(front_indices) > 0:
            if front_count >= len(front_indices):
                selected_indices.extend(front_indices)
                remaining = front_count - len(front_indices)
                side_count += remaining
            else:
                front_sample_indices = np.linspace(0, len(front_indices) - 1, front_count, dtype=int)
                selected_indices.extend(front_indices[front_sample_indices])
        
        # 側面エリアを低密度でサンプリング
        if len(side_indices) > 0 and side_count > 0:
            if side_count >= len(side_indices):
                selected_indices.extend(side_indices)
            else:
                side_sample_indices = np.linspace(0, len(side_indices) - 1, side_count, dtype=int)
                selected_indices.extend(side_indices[side_sample_indices])
        
        # 元の順序を維持するためにインデックスをソート
        selected_indices = np.sort(selected_indices)
        
        return scan_data[selected_indices]

    def _pad_frames(self, frames):
        """フレームをパディングする（必要に応じて拡張）"""
        if not frames:
            return []
        
        # 必要に応じてフレームを複製してパディング
        while len(frames) < self.scan_window_size:
            frames.append(frames[-1] if frames else np.array([]))
        
        return frames
    
    def convert_action(self, action, steer_range: float=1.0, speed_range: float=1.0):
        steer = action[0] * steer_range
        speed = (action[1] + 1) / 2 * speed_range
        speed = min(speed, speed_range)
        action = [steer, speed]
        return action

    def control_function(self, x):
        # 範囲外の値をクランプ
        if x < 0.14:
            x = 0.14
        if x > 0.2:
            x = 0.2
                                            
        if x <= 0.17:
        # 区間1: 0.14 ≤ x ≤ 0.17 → 0.02 ≤ y ≤ 0.17
            return 5 * x - 0.68
        else:
        # 区間2: 0.17 < x ≤ 0.2 → 0.17 < y ≤ 0.2
            return x
    def _downsample(self, frame):
        """単一フレームのダウンサンプリング"""
        if self.use_adaptive_downsampling:
            return self.downsample_single_frame(frame, self.downsample_num)
        else:
            # 従来の等間隔サンプリング
            if len(frame) == self.downsample_num:
                return frame
            indices = np.linspace(0, len(frame) - 1, self.downsample_num).astype(int)
            return frame[indices]

    def get_concatenated_numpy(self) -> np.ndarray:
        """
        フレームをNumPy配列として返す
        target_sizeが設定されている場合は適応的ダウンサンプリングを実行
        
        Returns:
            ダウンサンプリング済み配列
        """
        frames = list(self.scan_window)
        frames = self._pad_frames(frames)
        processed = [self._downsample(f) for f in frames]
        return np.hstack(processed)

    def scan_callback(self, msg):
        self.step_count += 1
        
        full_ranges = np.array(msg.ranges, dtype=np.float32)
        full_ranges = np.nan_to_num(full_ranges, nan=self.max_range, posinf=self.max_range)

        num_beams = len(full_ranges)
        if self.downsample_num > num_beams:
            self.get_logger().warn("downsample_num > number of beams in scan.")
            return

        # スキャンデータをウィンドウに追加
        self.scan_window.append(full_ranges)

        # ダウンサンプリング処理
        if self.use_adaptive_downsampling:
            sampled_ranges = self.downsample_single_frame(full_ranges, self.downsample_num)
        else:
            # 従来の等間隔サンプリング
            indices = np.linspace(0, num_beams - 1, self.downsample_num).astype(int)
            sampled_ranges = full_ranges[indices]

        # 🆕 カーブクラス推論の実行
        current_stable_result = None
        speed_factor = 1.0
        
        if self.inference_enabled and self.stabilized_inference:
            try:
                result = self.stabilized_inference.predict_and_stabilize(
                    scan_data=sampled_ranges,
                    current_step=self.step_count
                )
                classs = result['inference_result']['predicted_class']
                self.get_logger().info(f"{classs}")
                # 現在の安定した結果を取得
                current_stable_result = self.stabilized_inference.get_current_stable_class()
                
                # 安定した推論結果に基づく速度調整係数の計算
                if current_stable_result and current_stable_result['stable_result'] is not None:
                    curvature_class = current_stable_result['stable_class']
                    confidence = current_stable_result['stable_confidence']
                    
                    # 高信頼度の安定した結果に基づいて速度係数を調整
                    if confidence > 0.8:
                        if curvature_class == 2 or curvature_class == 1:  # 中程度のカーブ
                            speed_factor = 1.0
                        elif curvature_class == 3:  # 急カーブクラスの場合
                            speed_factor = 0.8
                        elif curvature_class == 0:  # 直線クラスの場合
                            speed_factor = 1.5
                
                # 推論結果の定期的な表示
                if self.step_count % self.inference_display_interval == 0:
                    if result['is_updated'] and result['stable_result']:
                        self.get_logger().info(
                            f"🎯 Curvature Inference [Step {self.step_count}]: "
                            f"Class={curvature_class}, Confidence={confidence:.3f}, "
                            f"Speed Factor={speed_factor:.2f}"
                        )
                    
            except Exception as e:
                self.get_logger().error(f"⚠️ Curvature inference error at step {self.step_count}: {e}")

        # LiDARデータをPyTorchテンソルに変換
        scan_tensor = torch.tensor(sampled_ranges, dtype=torch.float32).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                # SACモデルの決定論的推論：sample()の第3戻り値（平均アクション）を使用
                _, _, action = self.model.sample(scan_tensor)
                
                steer, throttle = action.tolist()[0]
                throttle = (throttle + 1.0) * 0.5

        except Exception as e:
            self.get_logger().error(f"SAC inference failed: {e}")
            return

        # 出力ゲインを適用
        steer *= self.output_steering_gain
        throttle *= self.output_throttle_gain
        throttle = self.control_function(throttle)*self.final_gain


        # 🆕 カーブクラス推論に基づく速度調整（オプション）
        if self.enable_speed_adjustment and self.inference_enabled:
            throttle *= speed_factor
            if self.step_count % self.inference_display_interval == 0 and speed_factor != 1.0:
                self.get_logger().info(f"🚗 Speed adjusted by curvature inference: {speed_factor:.2f}")

        # ゲイン適用後に値をクリッピングして安全な範囲に収める
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        

        drive_msg = AckermannDrive()
        drive_msg.steering_angle = steer
        drive_msg.speed = throttle
        self.publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 最終統計の表示
        if node.inference_enabled and node.stabilized_inference:
            try:
                final_stats = node.stabilized_inference.get_statistics()
                if final_stats:
                    node.get_logger().info("🏁 Final Curvature Inference Statistics:")
                    node.get_logger().info(f"   Total Predictions: {final_stats['total_predictions']}")
                    node.get_logger().info(f"   Stable Updates: {final_stats['stable_updates']}")
                    node.get_logger().info(f"   Stability Rate: {final_stats['stability_rate']:.1f}%")
                    node.get_logger().info(f"   Final Stable Class: {final_stats['current_stable_class']}")
            except:
                pass
        
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
