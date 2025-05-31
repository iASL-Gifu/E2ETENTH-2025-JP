import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from rcl_interfaces.msg import SetParametersResult

import torch
import numpy as np
from torch_geometric.data import Data

from .models.gnn import LidarGCN
import lidar_graph_cuda  # CUDAでビルドされた拡張モジュール


class GNNNode(Node):
    def __init__(self):
        super().__init__('lidar_gnn_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[DEVICE] Using device: {self.device}")

        self.declare_parameter('model_path', 'gnn_checkpoint.pt')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('downsample_num', 1081)
        self.declare_parameter('distance_threshold', 1.0)
        self.declare_parameter('model_input_dim', 2)
        self.declare_parameter('model_hidden_dim', 64)
        self.declare_parameter('model_output_dim', 2)


        self.load_parameters()

        self.model = self.load_model(self.model_path)

        self.add_on_set_parameters_callback(self.on_param_change)

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.publisher = self.create_publisher(
            AckermannDrive,
            '/cmd_drive',
            10)

        self.get_logger().info(f"[STARTED] model: {self.model_path}")

    def load_parameters(self):
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.downsample_num = self.get_parameter('downsample_num').get_parameter_value().integer_value
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value

        # モデル構造のパラメータ
        self.model_input_dim = self.get_parameter('model_input_dim').get_parameter_value().integer_value
        self.model_hidden_dim = self.get_parameter('model_hidden_dim').get_parameter_value().integer_value
        self.model_output_dim = self.get_parameter('model_output_dim').get_parameter_value().integer_value


    def on_param_change(self, params):
        success = True
        reason = ""
        reload_needed = False

        for param in params:
            if param.name == 'model_path':
                try:
                    self.model_path = param.value
                    reload_needed = True
                except Exception as e:
                    success = False
                    reason += f" Model path update failed: {e}"
            elif param.name == 'max_range':
                self.max_range = param.value
            elif param.name == 'downsample_num':
                self.downsample_num = param.value
            elif param.name == 'distance_threshold':
                self.distance_threshold = param.value
            elif param.name == 'model_input_dim':
                self.model_input_dim = param.value
                reload_needed = True
            elif param.name == 'model_hidden_dim':
                self.model_hidden_dim = param.value
                reload_needed = True
            elif param.name == 'model_output_dim':
                self.model_output_dim = param.value
                reload_needed = True

        if reload_needed:
            try:
                self.model = self.load_model(self.model_path)
                self.get_logger().info(f"[RELOADED] Model from {self.model_path}")
            except Exception as e:
                success = False
                reason += f" Model reload failed: {e}"

        return SetParametersResult(successful=success, reason=reason)

    def load_model(self, path):
        model = LidarGCN(
            input_dim=self.model_input_dim,
            hidden_dim=self.model_hidden_dim,
            output_dim=self.model_output_dim
        ).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model


    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.max_range, posinf=self.max_range)

        if self.downsample_num > len(ranges):
            self.get_logger().warn("downsample_num > number of beams in scan.")
            return

        indices = np.linspace(0, len(ranges) - 1, self.downsample_num).astype(int)
        sampled = ranges[indices]
        sampled = np.clip(sampled, 0.0, self.max_range)

        scan_tensor = torch.tensor(sampled / self.max_range, dtype=torch.float32).to(self.device)

        try:
            with torch.no_grad():
                edge_index, edge_attr, x, y = lidar_graph_cuda.build_lidar_graph_cuda(scan_tensor, self.distance_threshold)
                node_features = torch.stack([x, y], dim=1)

                data = Data(
                    x=node_features,
                    edge_index=edge_index.T.contiguous(),  # PyG expects [2, E]
                    edge_attr=edge_attr,
                    batch=torch.zeros(node_features.size(0), dtype=torch.long).to(self.device)
                )

                output = self.model(data)
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
    node = GNNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
