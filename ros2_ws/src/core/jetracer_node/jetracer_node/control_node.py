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
        self.declare_parameter('throttle_inversion', False) # スロットル反転フラグ
        self.declare_parameter('steering_inversion', False) # ステアリング反転フラグ
        self.declare_parameter('steering_offset', 0.0)      # ステアリングのオフセット
        self.declare_parameter('throttle_offset', 0.0)      # スロットルのオフセット
        self.declare_parameter('offset_step', 0.01)         # オフセットの増減量
        
        ### 変更点：ゲイン（比率）調整用のパラメータを追加 ###
        self.declare_parameter('throttle_gain', 1.0)        # スロットルゲイン
        self.declare_parameter('steering_gain', 1.0)        # ステアリングゲイン

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

    def _get_param(self, name: str, param_type: Parameter.Type = Parameter.Type.DOUBLE) -> any:
        param = self.get_parameter(name).get_parameter_value()
        if param_type == Parameter.Type.BOOL:
            return param.bool_value
        elif param_type == Parameter.Type.DOUBLE:
            return param.double_value
        
        return param.double_value 

    def _set_param(self, name: str, value: float):
        self.set_parameters([Parameter(name, Parameter.Type.DOUBLE, value)])
        self.get_logger().info(f'{name} updated to: {value:.3f}')

    def _cmd_cb(self, msg: AckermannDrive):
        # パラメータを取得
        steering_offset = self._get_param('steering_offset')
        throttle_offset = self._get_param('throttle_offset')
        throttle_inversion = self._get_param('throttle_inversion', Parameter.Type.BOOL)
        steering_inversion = self._get_param('steering_inversion', Parameter.Type.BOOL)
        throttle_gain = self._get_param('throttle_gain')
        steering_gain = self._get_param('steering_gain')

        # スロットル計算
        throttle = (msg.speed * throttle_gain) + throttle_offset
        if throttle_inversion:
            throttle *= -1.0
        throttle = max(min(throttle, 1.0), -1.0) # -1.0 から 1.0 の範囲にクリッピング

        # ステアリング計算
        steering = (msg.steering_angle * steering_gain) + steering_offset
        if steering_inversion:
            steering *= -1.0
        steering = max(min(steering, 1.0), -1.0) # -1.0 から 1.0 の範囲にクリッピング

        self.car.throttle = float(throttle)
        self.car.steering = float(steering)
        self.last_cmd_time = self.get_clock().now()

    def _watchdog(self):
        # 1秒以上コマンドが来ていなければスロットルとステアリングを0にする
        if (self.get_clock().now() - self.last_cmd_time).nanoseconds > 1e9: # 1秒 = 1e9ナノ秒
            self.car.throttle = 0.0
            self.car.steering = 0.0 

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