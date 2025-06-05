import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from rcl_interfaces.msg import SetParametersResult

import torch
import numpy as np
from torch_geometric.data import Data

# 学習コードと共通のモデルローダーをインポート
from .models.models import load_gnn_model
# CUDAでビルドされた拡張モジュール
import lidar_graph_cuda


class GNNNode(Node):
    def __init__(self):
        super().__init__('lidar_gnn_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[DEVICE] Using device: {self.device}")

        # モデル生成に必要なパラメータをすべて宣言
        self.declare_parameter('model_name', 'LidarGCN')
        self.declare_parameter('model_path', 'gnn_checkpoint.pt')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('downsample_num', 1081)
        self.declare_parameter('distance_threshold', 1.0)
        # モデル構造のパラメータ
        self.declare_parameter('input_dim', 2)
        self.declare_parameter('hidden_dim', 64)
        self.declare_parameter('output_dim', 2)
        # GNN/GAT用の追加パラメータ
        self.declare_parameter('pool_method', 'mean')
        self.declare_parameter('heads', 8)
        # LSTM用の追加パラメータ
        self.declare_parameter('lstm_hidden_dim', 128)

        # パラメータを読み込み
        self.load_parameters()

        # モデルを動的に読み込み
        self.model = self.load_model(self.model_path)

        # パラメータ変更時のコールバックを登録
        self.add_on_set_parameters_callback(self.on_param_change)

        # Subscriber / Publisher の設定
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.publisher = self.create_publisher(
            AckermannDrive,
            '/cmd_drive',
            10)

        self.get_logger().info(f"[STARTED] model: {self.model_name} from {self.model_path}")

    def load_parameters(self):
        """ROSパラメータを読み込み、self変数に格納する"""
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.downsample_num = self.get_parameter('downsample_num').get_parameter_value().integer_value
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        
        # モデル構造のパラメータ
        self.model_input_dim = self.get_parameter('input_dim').get_parameter_value().integer_value
        self.model_hidden_dim = self.get_parameter('hidden_dim').get_parameter_value().integer_value
        self.model_output_dim = self.get_parameter('output_dim').get_parameter_value().integer_value
        self.model_pool_method = self.get_parameter('pool_method').get_parameter_value().string_value
        self.model_heads = self.get_parameter('heads').get_parameter_value().integer_value
        self.model_lstm_hidden_dim = self.get_parameter('lstm_hidden_dim').get_parameter_value().integer_value

    def on_param_change(self, params):
        """パラメータ動的変更時の処理"""
        success = True
        reason = ""
        reload_model_needed = any(p.name in [
            'model_name', 'model_path', 'input_dim', 'hidden_dim', 
            'output_dim', 'pool_method', 'heads', 'lstm_hidden_dim'
        ] for p in params)
        
        self.load_parameters()

        if reload_model_needed:
            try:
                self.model = self.load_model(self.model_path)
                self.get_logger().info(f"[RELOADED] Model: {self.model_name} from {self.model_path}")
            except Exception as e:
                success = False
                reason = f"Model reload failed: {e}"

        return SetParametersResult(successful=success, reason=reason)

    def load_model(self, path):
        """load_gnn_model関数を使ってモデルを生成・読み込みする"""
        model = load_gnn_model(
            model_name=self.model_name,
            input_dim=self.model_input_dim,
            hidden_dim=self.model_hidden_dim,
            output_dim=self.model_output_dim,
            pool_method=self.model_pool_method,
        ).to(self.device)
        
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def scan_callback(self, msg):
        # 1. スキャンデータの前処理
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.max_range, posinf=self.max_range)
        if self.downsample_num > len(ranges):
            self.get_logger().warn(f"downsample_num({self.downsample_num}) > number of beams({len(ranges)}) in scan.")
            return
        indices = np.linspace(0, len(ranges) - 1, self.downsample_num).astype(int)
        sampled = ranges[indices]
        scan_tensor = torch.tensor(sampled, dtype=torch.float32).to(self.device)

        try:
            with torch.no_grad():
                # 2. グラフ構築
                edge_index, edge_attr, x, y = lidar_graph_cuda.build_lidar_graph_cuda(
                    scan_tensor, self.distance_threshold, max_neighbors=5
                )
                
                mask = edge_index[:, 0] >= 0
                edge_index = edge_index[mask].t().contiguous()
                edge_attr = edge_attr[mask]

                node_features = torch.stack([x, y], dim=1)
                data = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=torch.zeros(node_features.size(0), dtype=torch.long).to(self.device)
                )

                # ★★★ 修正箇所: 常にリストでモデルに入力 ★★★
                # 全てのモデルがリスト入力を処理できるため、分岐が不要になった
                model_input = [data]

                # 4. 推論実行
                output = self.model(model_input)
                steer, throttle = output[0].tolist()

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}", exc_info=True)
            return

        # 5. 制御命令の送信
        drive_msg = AckermannDrive()
        drive_msg.steering_angle = float(np.clip(steer, -1.0, 1.0))
        drive_msg.speed = float(np.clip(throttle, -0.5, 0.5))
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