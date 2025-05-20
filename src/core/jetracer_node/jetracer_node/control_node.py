#!/usr/bin/env python3
"""
ROS 2 → JetRacer ブリッジ。
/cmd_drive (ackermann_msgs/AckermannDrive) を受け取り
NvidiaRacecar に throttle, steering を流し込むだけ。
オフセット付き & 外部トピックからの動的調整対応
"""
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Bool
from jetracer.nvidia_racecar import NvidiaRacecar
from rclpy.qos import QoSProfile
from rclpy.parameter import Parameter


class JetRacerDriver(Node):
    def __init__(self):
        super().__init__('jetracer_driver')

        # パラメータ宣言（YAML設定ファイル対応）
        self.declare_parameter('steering_offset', 0.0)
        self.declare_parameter('throttle_offset', 0.0)
        self.declare_parameter('offset_step', 0.01)  # 増減量をパラメータに

        self.car = NvidiaRacecar()
        self.last_cmd_time = self.get_clock().now()
        self.car.throttle = 0.0
        self.car.steering = 0.0

        qos = QoSProfile(depth=10)

        # コマンド購読
        self.create_subscription(
            AckermannDrive,
            '/cmd_drive',
            self._cmd_cb,
            qos_profile=qos
        )

        # オフセット調整トピック
        self.create_subscription(Bool, '/steer_offset_inc', self._steer_offset_inc_cb, qos)
        self.create_subscription(Bool, '/steer_offset_dec', self._steer_offset_dec_cb, qos)
        self.create_subscription(Bool, '/speed_offset_inc', self._speed_offset_inc_cb, qos)
        self.create_subscription(Bool, '/speed_offset_dec', self._speed_offset_dec_cb, qos)

        self.create_timer(0.1, self._watchdog)
        self.get_logger().info('JetRacer driver started, waiting for /cmd_drive')

    def _get_param(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _set_param(self, name: str, value: float):
        self.set_parameters([Parameter(name, Parameter.Type.DOUBLE, value)])
        self.get_logger().info(f'{name} updated to: {value:.3f}')

    def _cmd_cb(self, msg: AckermannDrive):
        steering_offset = self._get_param('steering_offset')
        throttle_offset = self._get_param('throttle_offset')

        throttle = max(min(msg.speed + throttle_offset, 1.0), -1.0)
        steering = max(min(msg.steering_angle + steering_offset, 1.0), -1.0)

        self.car.throttle = float(throttle)
        self.car.steering = float(steering)
        self.last_cmd_time = self.get_clock().now()

    def _watchdog(self):
        if (self.get_clock().now() - self.last_cmd_time).nanoseconds > 1e9:
            self.car.throttle = 0.0
            self.car.steer = 0.0

    # --- 動的調整コールバック ---
    def _steer_offset_inc_cb(self, msg: Bool):
        if msg.data:
            step = self._get_param('offset_step')
            current = self._get_param('steering_offset')
            self._set_param('steering_offset', current + step)

    def _steer_offset_dec_cb(self, msg: Bool):
        if msg.data:
            step = self._get_param('offset_step')
            current = self._get_param('steering_offset')
            self._set_param('steering_offset', current - step)

    def _speed_offset_inc_cb(self, msg: Bool):
        if msg.data:
            step = self._get_param('offset_step')
            current = self._get_param('throttle_offset')
            self._set_param('throttle_offset', current + step)

    def _speed_offset_dec_cb(self, msg: Bool):
        if msg.data:
            step = self._get_param('offset_step')
            current = self._get_param('throttle_offset')
            self._set_param('throttle_offset', current - step)


def main():
    rclpy.init()
    node = JetRacerDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
