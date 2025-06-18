import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from rcl_interfaces.msg import SetParametersResult
import torch
import numpy as np
import os
from collections import deque

# ファクトリ関数をインポート
from .models.models import load_cnn_model 

class CNNNode(Node):
    def __init__(self):
        super().__init__('lidar_drive_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[DEVICE] Using device: {self.device}")

        # --- パラメータ宣言 ---
        self.declare_parameter('model_name', 'TinyLidarNet')
        self.declare_parameter('model_path', 'model.pth')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('input_dim', 181) # 各スキャンフレームのLiDARデータ次元
        self.declare_parameter('output_dim', 2)
        self.declare_parameter('sequence_length', 1) 

        # --- 状態変数の初期化 ---
        self.model = None
        self.hidden_state = None
        self.prev_action = None
        self.is_rnn = False
        self.use_prev_action = False
        
        # ▼▼▼ 変更点: scan_bufferのmaxlenはload_parametersで設定されるため、最初は0で初期化 ▼▼▼
        self.scan_buffer = deque(maxlen=0) 

        # --- 初期設定の実行 ---
        self.load_parameters() # ここで sequence_length から buffer_size が設定され、scan_bufferが初期化される
        self.model = self.load_and_prepare_model()

        # パラメータ変更時のコールバック登録
        self.add_on_set_parameters_callback(self.on_param_change)

        # ROS I/O
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(AckermannDrive, '/cmd_drive', 10)

        self.get_logger().info(f"[STARTED] Node is ready. Model: {self.model_name}")
        self.get_logger().info(f"  > Derived flags: IsRNN={self.is_rnn}, UsePrevAction={self.use_prev_action}")
        self.get_logger().info(f"  > Sequence Length: {self.sequence_length}")


    def load_parameters(self):
        """ROSパラメータを読み込み、フラグをモデル名から派生させる"""
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.input_dim = self.get_parameter('input_dim').get_parameter_value().integer_value
        self.output_dim = self.get_parameter('output_dim').get_parameter_value().integer_value
        
        # ▼▼▼ 変更点: buffer_size を sequence_length と同じ値に設定 ▼▼▼
        old_sequence_length = self.sequence_length if hasattr(self, 'sequence_length') else -1
        self.sequence_length = self.get_parameter('sequence_length').get_parameter_value().integer_value
        self.buffer_size = self.sequence_length # buffer_size は sequence_length と同じ

        # model_name に基づいてフラグを判定
        self.is_rnn = "Lstm" in self.model_name
        self.use_prev_action = "Action" in self.model_name
        
        # 初回のみ prev_action を初期化
        if self.prev_action is None:
             self.prev_action = np.zeros(self.output_dim, dtype=np.float32)

        # ▼▼▼ 変更点: sequence_length の変更時にバッファをリセット ▼▼▼
        if old_sequence_length != self.sequence_length:
            self.get_logger().info(f"Sequence length changed from {old_sequence_length} to {self.sequence_length}. Resetting scan buffer.")
            self.scan_buffer = deque(maxlen=self.buffer_size) # 新しいmaxlenでdequeを再作成し、バッファをクリア
        elif self.scan_buffer.maxlen != self.buffer_size:
            # maxlenが何らかの理由で異なる場合は再初期化
            self.scan_buffer = deque(maxlen=self.buffer_size)
            self.get_logger().info(f"Scan buffer maxlen adjusted to {self.buffer_size}")


    def on_param_change(self, params):
        """パラメータの動的変更をハンドルし、必要ならモデルをリロード"""
        # モデル構造に関わるパラメータが変更されたかどうかのフラグ
        reload_needed = any(p.name in ['model_name', 'model_path', 'input_dim', 'output_dim'] for p in params)
        # sequence_length に関するパラメータが変更されたかどうかのフラグ
        sequence_length_changed = any(p.name == 'sequence_length' for p in params)
        
        # 現在の sequence_length を保存 (変更前の値と比較するため)
        old_sequence_length = self.sequence_length

        # 実際にパラメータをクラス変数に反映
        for param in params:
            if hasattr(self, param.name):
                # sequence_length が変更された場合のみ、それに合わせて buffer_size も更新
                if param.name == 'sequence_length':
                    setattr(self, param.name, param.value)
                    self.buffer_size = self.sequence_length # buffer_size を sequence_length と同期
                else:
                    setattr(self, param.name, param.value)
        
        # model_name が変更された可能性があるため、フラグを再評価
        self.is_rnn = "Lstm" in self.model_name
        self.use_prev_action = "Action" in self.model_name

        # ▼▼▼ 変更点: sequence_length の変更時にバッファをリセット ▼▼▼
        if sequence_length_changed and old_sequence_length != self.sequence_length:
            self.get_logger().info(f"Sequence length changed from {old_sequence_length} to {self.sequence_length}. Resetting scan buffer.")
            self.scan_buffer = deque(maxlen=self.buffer_size) # 新しいmaxlenでdequeを再作成し、バッファをクリア
            self.get_logger().info(f"  > New sequence length: {self.sequence_length}")


        if reload_needed:
            self.get_logger().info("Model-related parameter changed. Reloading model...")
            self.model = self.load_and_prepare_model()
            if self.model is None:
                return SetParametersResult(successful=False, reason="Model reload failed.")
            self.get_logger().info(f"Model reloaded. New model: {self.model_name}")
            self.get_logger().info(f"  > New derived flags: IsRNN={self.is_rnn}, UsePrevAction={self.use_prev_action}")

        return SetParametersResult(successful=True)
    
    def load_and_prepare_model(self):
        """モデルをインスタンス化し、学習済み重みをロードして準備する"""
        try:
            self.get_logger().info(
                f"Loading model '{self.model_name}' with input_dim={self.input_dim}, output_dim={self.output_dim}"
            )
            
            # sequence_lengthをモデルのロード時に渡す必要がある場合、load_cnn_modelを修正する必要があります。
            # 例えば、Transformerモデルの入力シーケンス長を指定する場合など。
            # ここでは、model_nameから自動的に内部で処理されると仮定します。
            model = load_cnn_model(
                model_name=self.model_name,
                input_dim=self.input_dim,
                output_dim=self.output_dim
            )
            
            if not os.path.exists(self.model_path):
                self.get_logger().warn(f"Model weight file not found: {self.model_path}. Using initial weights.")
            else:
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.get_logger().info(f"Loaded weights from {self.model_path}")

            model.eval()
            model.to(self.device)
            
            self.get_logger().info("Resetting RNN hidden state and previous action.")
            self.hidden_state = None
            self.prev_action = np.zeros(self.output_dim, dtype=np.float32)
            # モデルリロード時にバッファもクリア
            self.scan_buffer.clear()
            
            return model
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return None
        
    def scan_callback(self, msg):
        if self.model is None:
            self.get_logger().warn("Model is not loaded, skipping inference.", throttle_skip_first=True, throttle_time_sec=5.0)
            return
            
        # --- 1. LiDARデータを準備し、バッファに追加 ---
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
        
        # バッファに現在のスキャンデータを追加
        self.scan_buffer.append(sampled_ranges)

        # バッファに十分なデータがない場合は推論をスキップ
        if len(self.scan_buffer) < self.sequence_length:
            self.get_logger().debug(f"Buffer has only {len(self.scan_buffer)} frames, waiting for {self.sequence_length} frames.")
            return # 推論に必要なデータが揃うまで待つ

        # 推論に使うフレームをバッファから取得
        # 末尾から sequence_length だけ取り出す (最新のデータから)
        # scan_sequence: [sequence_length, input_dim]
        scan_sequence = np.array(list(self.scan_buffer), dtype=np.float32) # maxlenがsequence_lengthと同じなので、list(self.scan_buffer)で全て取得

        # --- 2. 推論の実行 ---
        try:
            with torch.no_grad():
                # --- 入力テンソルを準備 ---
                # scan_tensor: [B, T, L] (バッチサイズ, シーケンス長, 特徴量次元)
                # ここではバッチサイズ1なので [1, sequence_length, input_dim]
                scan_tensor = torch.tensor(scan_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                prev_action_tensor = None
                if self.use_prev_action:
                    # prev_action_tensor: [B, A] または [B, T, A] (モデルの入力による)
                    prev_action_tensor = torch.tensor(self.prev_action, dtype=torch.float32).unsqueeze(0).to(self.device)

                output = None
                
                # --- 3. モデルのタイプに応じて形状を調整し、推論 ---
                if self.is_rnn:
                    # === RNN (LSTM/ConvLSTM) モデルの場合 ===
                    if self.hidden_state is not None:
                        if isinstance(self.hidden_state, tuple):
                            self.hidden_state = tuple(h.detach() for h in self.hidden_state)
                        else:
                            self.hidden_state = self.hidden_state.detach()
                    
                    if self.use_prev_action:
                        output, self.hidden_state = self.model(scan_tensor, prev_action_tensor, self.hidden_state)
                    else:
                        output, self.hidden_state = self.model(scan_tensor, self.hidden_state)
                else:
                    # === 非RNN (CNN, Transformerなど) モデルの場合 ===
                    # CNNが Conv1D の場合 [B, C, L] に変換が必要な可能性あり
                    # Transformerは [B, T, L] の形式を受け入れることが多い
                    # ここでは scan_tensor が [1, sequence_length, input_dim] のため、そのまま渡す
                    
                    if self.use_prev_action:
                        output = self.model(scan_tensor, prev_action_tensor)
                    else:
                        output = self.model(scan_tensor)
                
                # --- 4. 結果の処理 ---
                self.prev_action = output[0].cpu().numpy()
                steer, throttle = self.prev_action.tolist()

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}", throttle_skip_first=True, throttle_time_sec=5.0)
            # エラー発生時は状態をリセット
            self.hidden_state = None
            if self.prev_action is not None:
                self.prev_action.fill(0)
            return
            
        drive_msg = AckermannDrive()
        drive_msg.steering_angle = float(steer)
        drive_msg.speed = float(throttle)
        self.publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CNNNode()
    
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