import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from rcl_interfaces.msg import SetParametersResult
import torch
import numpy as np
import os

from .models.actor import get_actor  # モデル定義

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        # デバイス設定（CUDA 優先）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[DEVICE] Using device: {self.device}")

        # パラメータ宣言
        self.declare_parameter('ckpt_path', 'checkpoint.pt')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('downsample_num', 1080)
        self.declare_parameter('state_dim', 1080)
        self.declare_parameter('action_dim', 2)
        self.declare_parameter('hidden_dim', 256)
        self.declare_parameter('policy_type', 'mlp')
        self.declare_parameter('output_steering_gain', 1.0)
        self.declare_parameter('output_throttle_gain', 1.0)


        # パラメータの読み込み
        self.load_parameters()

        # モデルをデバイス上に配置
        self.model = self.load_model(self.ckpt_path,
                                     self.state_dim,
                                     self.action_dim,
                                     self.hidden_dim,
                                     self.policy_type)

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
            f"[STARTED] model: {self.ckpt_path}, downsample_num: {self.downsample_num}, max_range: {self.max_range}"
        )

    def load_parameters(self):
        self.ckpt_path = self.get_parameter('ckpt_path').get_parameter_value().string_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.downsample_num = self.get_parameter('downsample_num').get_parameter_value().integer_value
        self.state_dim = self.get_parameter('state_dim').get_parameter_value().integer_value
        self.action_dim = self.get_parameter('action_dim').get_parameter_value().integer_value
        self.hidden_dim = self.get_parameter('hidden_dim').get_parameter_value().integer_value
        self.policy_type = self.get_parameter('policy_type').get_parameter_value().string_value
        ### 変更点：ゲインパラメータを読み込む ###
        self.output_steering_gain = self.get_parameter('output_steering_gain').get_parameter_value().double_value
        self.output_throttle_gain = self.get_parameter('output_throttle_gain').get_parameter_value().double_value


    def on_param_change(self, params):
        success = True
        reason = ""

        for param in params:
            if param.name == 'ckpt_path':
                try:
                    self.model = self.load_model(param.value,
                                                 self.state_dim,
                                                 self.action_dim,
                                                 self.hidden_dim,
                                                 self.policy_type)
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
            elif param.name == 'state_dim':
                self.state_dim = param.value
            elif param.name == 'action_dim':
                self.action_dim = param.value
            elif param.name == 'hidden_dim':
                self.hidden_dim = param.value
            elif param.name == 'policy_type':
                self.policy_type = param.value
            
            ### 変更点：ゲインパラメータの動的変更に対応 ###
            elif param.name == 'output_steering_gain':
                self.output_steering_gain = param.value
                self.get_logger().info(f"[UPDATED] output_steering_gain: {self.output_steering_gain}")
            elif param.name == 'output_throttle_gain':
                self.output_throttle_gain = param.value
                self.get_logger().info(f"[UPDATED] output_throttle_gain: {self.output_throttle_gain}")

        # state_dim, action_dim, hidden_dim, policy_type はモデル再読み込み時に使われるため、
        # ログ出力のみにとどめ、値の更新のみ行う
        if any(p.name in ['state_dim', 'action_dim', 'hidden_dim', 'policy_type'] for p in params):
            self.get_logger().info("Model architecture parameters updated. Reload model (change ckpt_path) to apply.")

        return SetParametersResult(successful=success, reason=reason)

    def load_model(self, path, state_dim, action_dim, hidden_dim, policy_type):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model = get_actor(state_dim=state_dim,
                          action_dim=action_dim,
                          hidden_dim=hidden_dim,
                          policy_type=policy_type)
        checkpoint = torch.load(path, map_location=self.device)
            
        if 'actor' in checkpoint:
            model.load_state_dict(checkpoint['actor'])
            self.get_logger().info(f"[✔] Pretrained actor weights successfully loaded from {path}")
        model.eval()
        model.to(self.device)
        return model

    def scan_callback(self, msg):
        full_ranges = np.array(msg.ranges, dtype=np.float32)
        full_ranges = np.nan_to_num(full_ranges, nan=self.max_range, posinf=self.max_range)

        num_beams = len(full_ranges)
        if self.downsample_num > num_beams:
            self.get_logger().warn("downsample_num > number of beams in scan.")
            return

        indices = np.linspace(0, num_beams - 1, self.downsample_num).astype(int)
        sampled_ranges = full_ranges[indices]

        sampled_ranges = np.clip(sampled_ranges, 0.0, self.max_range) / self.max_range
        scan_tensor = torch.tensor(sampled_ranges, dtype=torch.float32).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                mean, _ = self.model(scan_tensor)
                action = torch.tanh(mean)
                steer, throttle = action.tolist()[0]
                throttle = (throttle + 1.0) / 2.0
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