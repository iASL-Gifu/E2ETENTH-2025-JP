import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from rcl_interfaces.msg import SetParametersResult
import torch
import numpy as np
import os

from .models.cnn import TinyLidarNet  # ← モデル定義を忘れずに import


class CNNNode(Node):
    def __init__(self):
        super().__init__('lidar_drive_node')

        # Device設定（CUDA優先）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[DEVICE] Using device: {self.device}")

        # Declare parameters
        self.declare_parameter('model_path', 'checkpoint.pt')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('downsample_num', 1081)

        # Load parameters
        self.load_parameters()

        # Load model to device
        self.model = self.load_model(self.model_path)

        # Register dynamic param callback
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
            f"[STARTED] model: {self.model_path}, downsample_num: {self.downsample_num}, max_range: {self.max_range}"
        )

    def load_parameters(self):
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.downsample_num = self.get_parameter('downsample_num').get_parameter_value().integer_value

    def on_param_change(self, params):
        success = True
        reason = ""

        for param in params:
            if param.name == 'model_path':
                try:
                    self.model = self.load_model(param.value)
                    self.model_path = param.value
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

        return SetParametersResult(successful=success, reason=reason)

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model = TinyLidarNet(input_length=self.downsample_num)  # ここはユーザーのCNNアーキテクチャに合わせて変更
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
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
                output = self.model(scan_tensor)
                steer, throttle = output[0].tolist()
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))

        drive_msg = AckermannDrive()
        drive_msg.steering_angle = steer
        drive_msg.speed = throttle
        self.publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CNNNode()
    
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()