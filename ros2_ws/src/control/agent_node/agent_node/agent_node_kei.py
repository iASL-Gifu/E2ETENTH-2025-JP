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
        self.declare_parameter('use_adaptive_downsampling', True)  # 新しいパラメータ
        self.declare_parameter('scan_window_size', 1)  # フレーム履歴用

        # パラメータの読み込み
        self.load_parameters()

        # スキャンデータの履歴を保持するためのウィンドウ
        self.scan_window = deque(maxlen=self.scan_window_size)

        # モデルをデバイス上に配置
        self.model = self.load_model(self.ckpt_path,
                                     self.lidar_dim,
                                     self.action_dim,
                                     self.hidden_dim)

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
            f"[STARTED] SAC model: {self.ckpt_path}, downsample_num: {self.downsample_num}, max_range: {self.max_range}, adaptive: {self.use_adaptive_downsampling}"
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
        self.scan_window_size = self.get_parameter('scan_window_size').get_parameter_value().integer_value

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
            elif param.name == 'scan_window_size':
                self.scan_window_size = param.value
                self.scan_window = deque(maxlen=self.scan_window_size)
                self.get_logger().info(f"[UPDATED] scan_window_size: {self.scan_window_size}")

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
        
        例:
            # 1080ビーム（-135°～+135°）のLiDARデータを100点にダウンサンプリング
            original_scan = np.array([...])  # 1080個の距離値
            downsampled = downsample_single_frame(original_scan, 100)  # 100点（前方70点、側面30点）
            # 前方±30度エリア（240ビーム）が高密度、側面エリア（840ビーム）が低密度でサンプリングされる
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
        # 1080ビーム中、前方240ビーム（±30度）に重点を置く
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
    
    def convert_action(self,action, steer_range: float=1.0, speed_range: float=1.0):
    
        steer = action[0] * steer_range
        speed = (action[1] + 1) / 2 * speed_range
        speed = min(speed, speed_range)
        action = [steer, speed]
        print(action)
        return action

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

        # LiDARデータをPyTorchテンソルに変換
        scan_tensor = torch.tensor(sampled_ranges, dtype=torch.float32).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                # SACモデルの決定論的推論：sample()の第3戻り値（平均アクション）を使用
                _, _, action = self.model.sample(scan_tensor)
                
                ## cv_action = self.convert_action(action,)

                steer, throttle = action.tolist()[0]
                throttle = (throttle+1.0)*0.5

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        # 出力ゲインを適用
        steer *= self.output_steering_gain
        throttle *= self.output_throttle_gain

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
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
