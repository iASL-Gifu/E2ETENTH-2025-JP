#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import subprocess
import datetime
import signal
import os
import threading

class RosBagManagerNode(Node):
    def __init__(self):
        super().__init__('ros2_bag_manager_node')

        # パラメータ宣言・取得
        self.declare_parameter('output_dir', 'rosbag2_output')
        self.declare_parameter('all_topics', True)
        self.declare_parameter('topics', ['/rosbag2_recorder/trigger'])
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.all_topics = self.get_parameter('all_topics').get_parameter_value().bool_value
        self.topics = list(self.get_parameter('topics').get_parameter_value().string_array_value)

        # セッション用ディレクトリを作成（ノード起動時）
        now = datetime.datetime.now()
        session_ts = now.strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.output_dir, session_ts)
        os.makedirs(self.session_dir, exist_ok=True)
        self.get_logger().info(f"セッションディレクトリ作成: {self.session_dir}")

        # トリガー購読
        self.trigger_sub = self.create_subscription(
            Bool,
            '/rosbag2_recorder/trigger',
            self.trigger_callback,
            10
        )

        # プロセスと状態管理
        self.recording_process = None
        self.is_recording = False
        self.lock = threading.Lock()

    def trigger_callback(self, msg: Bool):
        with self.lock:  # ロックで排他制御
            if msg.data:
                # === 録画開始 ===
                if not self.is_recording:
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    record_dir = os.path.join(self.session_dir, ts)
                    # os.makedirs(record_dir, exist_ok=True)

                    cmd = ['ros2', 'bag', 'record']
                    if self.all_topics:
                        cmd.append('-a')
                    elif self.topics:
                        cmd.extend(self.topics)
                    else:
                        self.get_logger().warn("all_topics=false かつ topics=[] のため、何も録画しません")
                        return

                    cmd.extend(['-o', record_dir, '-s', 'mcap'])

                    # 追加: 実行コマンドの出力
                    command_str = ' '.join(cmd)
                    print(f"[DEBUG] Command to be executed:\n{command_str}")
                    self.get_logger().info(f"[DEBUG] Command to be executed:\n{command_str}")

                    self.get_logger().info(f"録画開始: {' '.join(cmd)}")
                    self.recording_process = subprocess.Popen(cmd, preexec_fn=os.setsid)
                    self.is_recording = True
                else:
                    self.get_logger().warn("すでに録画中です。新しい録画は開始しません。")
            else:
                # === 録画停止 ===
                if self.is_recording:
                    self.get_logger().info("録画停止要求 — SIGINT を送信します")
                    self._cleanup()
                else:
                    self.get_logger().warn("録画は現在行われていません。停止要求は無視されます。")

    def _cleanup(self):
        """録画プロセスが生きていたら SIGINT で停止し、wait する"""
        if self.recording_process:
            try:
                os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
                self.recording_process.wait(timeout=5)
            except Exception as e:
                self.get_logger().error(f"録画プロセス停止中にエラー: {e}")
            finally:
                self.recording_process = None
                self.is_recording = False
                self.get_logger().info("録画を停止しました")

    def destroy_node(self):
        # ノード終了時のクリーンアップ
        self.get_logger().info("ノードシャットダウン: 録画プロセス停止処理を実行します")
        self._cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RosBagManagerNode()

    # シグナルハンドラ登録（Ctrl+C や docker stop 等）
    def _signal_handler(signum, frame):
        node.get_logger().info(f"シグナル {signal.Signals(signum).name} 受信 — シャットダウンします")
        rclpy.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

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
