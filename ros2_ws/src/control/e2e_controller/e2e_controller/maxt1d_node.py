import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from rcl_interfaces.msg import SetParametersResult
import torch
import numpy as np
import os
import time 

# ファクトリ関数をインポート
from .models.models import load_maxt_model

class MAXT1dNode(Node):
    def __init__(self):
        super().__init__('maxt1d_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[DEVICE] Using device: {self.device}")

        # --- パラメータ宣言 ---
        self.declare_parameter('model_size', 'Tiny')
        self.declare_parameter('model_path', 'model.pth')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('input_dim', 181)
        self.declare_parameter('output_dim', 2)
        self.declare_parameter('backbone_stage', 'all')
        self.declare_parameter('neck_stage', 'all')
        self.declare_parameter('debug', False) 

        # --- 状態変数の初期化 ---
        self.model = None
        
        # --- 初期設定の実行 ---
        self.load_parameters()
        self.model = self.load_and_prepare_model()

        # パラメータ変更時のコールバック登録
        self.add_on_set_parameters_callback(self.on_param_change)

        # ROS I/O
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(AckermannDrive, '/cmd_drive', 10)

        self.get_logger().info(f"[STARTED] Node is ready. Model Size: {self.model_size}, Backbone: {self.backbone_stage}, Neck: {self.neck_stage}")

    def load_parameters(self):
        """ROSパラメータを読み込む"""
        self.model_size = self.get_parameter('model_size').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.input_dim = self.get_parameter('input_dim').get_parameter_value().integer_value
        self.output_dim = self.get_parameter('output_dim').get_parameter_value().integer_value
        self.backbone_stage = self.get_parameter('backbone_stage').get_parameter_value().string_value
        self.neck_stage = self.get_parameter('neck_stage').get_parameter_value().string_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value

    def on_param_change(self, params):
        """パラメータの動的変更をハンドルし、必要ならモデルをリロード"""
        reload_needed = any(p.name in ['model_size', 'model_path', 'input_dim', 'output_dim', 'backbone_stage', 'neck_stage'] for p in params)
        
        # 実際にパラメータをクラス変数に反映
        for param in params:
            if hasattr(self, param.name):
                # パラメータの型に応じて値を取得するように修正
                param_type = type(getattr(self, param.name))
                if param_type == bool:
                    setattr(self, param.name, param.value)
                elif param_type == int:
                    setattr(self, param.name, param.value)
                elif param_type == float:
                     setattr(self, param.name, param.value)
                else: # stringなど
                    setattr(self, param.name, param.value)

        if reload_needed:
            self.get_logger().info("Model-related parameter changed. Reloading model...")
            self.model = self.load_and_prepare_model()
            if self.model is None:
                self.get_logger().error("Model reload failed.")
                return SetParametersResult(successful=False, reason="Model reload failed.")
            self.get_logger().info(f"Model reloaded. New model size: {self.model_size}, Backbone: {self.backbone_stage}, Neck: {self.neck_stage}")
        
        
        for param in params:
            if param.name == 'debug':
                self.get_logger().info(f"Debug mode set to: {self.debug}")

        return SetParametersResult(successful=True)
    
    def load_and_prepare_model(self):
        """モデルをインスタンス化し、学習済み重みをロードして準備する"""
        try:
            # --- 変更: ログにbackbone_stageとneck_stageを追加 ---
            self.get_logger().info(
                f"Loading model '{self.model_size}' with: \n"
                f"  - input_dim: {self.input_dim}\n"
                f"  - output_dim: {self.output_dim}\n"
                f"  - backbone_stage: '{self.backbone_stage}'\n"
                f"  - neck_stage: '{self.neck_stage}'"
            )
            
            model = load_maxt_model(
                size=self.model_size,
                backbone_stage=self.backbone_stage,
                neck_stage=self.neck_stage
            )
            
            if not os.path.exists(self.model_path):
                self.get_logger().warn(f"Model weight file not found: {self.model_path}. Using initial weights.")
            else:
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.get_logger().info(f"Loaded weights from {self.model_path}")

            model.eval()
            model.to(self.device)
            
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return None
        
    def scan_callback(self, msg):
        if self.model is None:
            self.get_logger().warn("Model is not loaded, skipping inference.", throttle_skip_first=True, throttle_time_sec=5.0)
            return
            
        # --- 1. LiDARデータを準備 ---
        full_ranges = np.array(msg.ranges, dtype=np.float32)
        full_ranges = np.nan_to_num(full_ranges, nan=self.max_range, posinf=self.max_range)
        num_beams = len(full_ranges)
        
        if self.input_dim > num_beams:
            self.get_logger().warn(f"input_dim({self.input_dim}) > num_beams({num_beams}).", throttle_skip_first=True, throttle_time_sec=5.0)
            return
        
        # 指定された input_dim にサンプリング
        indices = np.linspace(0, num_beams - 1, self.input_dim).astype(int)
        sampled_ranges = full_ranges[indices]
        sampled_ranges = np.clip(sampled_ranges, 0.0, self.max_range) / self.max_range
        
        # --- 2. 推論の実行 ---
        try:
            with torch.no_grad():
                # --- 入力テンソルを準備 ---
                # scan_tensor: [B, C, L] (バッチサイズ, チャンネル, 特徴量次元)
                # ここではバッチサイズ1, チャンネル1なので [1, 1, input_dim]
                scan_tensor = torch.tensor(sampled_ranges, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                
                if self.debug:
                    start_time = time.perf_counter()

                # --- 3. モデルで推論 ---
                result_dict = self.model(scan_tensor)

                if self.debug:
                    end_time = time.perf_counter()
                    inference_time_ms = (end_time - start_time) * 1000
                    self.get_logger().info(f"Inference Time: {inference_time_ms:.3f} ms")
                
                # --- 4. 結果の処理 ---
                predictions_dict = result_dict['output']
                if predictions_dict:
                    # テンソルのリストをスタックし(num_scales, 1, 2)、dim=0で平均を取る -> (1, 2)
                    avg_prediction_tensor = torch.stack(list(predictions_dict.values())).mean(dim=0)
                    self.prev_action = avg_prediction_tensor[0].cpu().numpy()
                    steer, throttle = self.prev_action.tolist()

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}", throttle_skip_first=True, throttle_time_sec=5.0)
            return
            
        drive_msg = AckermannDrive()
        drive_msg.steering_angle = float(steer)
        drive_msg.speed = float(throttle)
        self.publisher.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    # クラス名に合わせてインスタンス化
    node = MAXT1dNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down.")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()