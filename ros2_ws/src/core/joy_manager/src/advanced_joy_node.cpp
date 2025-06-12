#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "ackermann_msgs/msg/ackermann_drive.hpp"
#include "std_msgs/msg/bool.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

class AdvancedJoyManagerNode : public rclcpp::Node
{
public:
  AdvancedJoyManagerNode()
  : Node("advanced_joy_manager_node"),
    joy_enable_pressed_(false),
    steer_gain_pressed_(false),
    autonomy_pressed_(false),
    raw_speed_(0.0),
    raw_steer_(0.0),
    prev_start_pressed_(false),
    prev_stop_pressed_(false),
    prev_steer_scale_inc_pressed_(false),
    prev_steer_scale_dec_pressed_(false),
    prev_speed_scale_inc_pressed_(false),
    prev_speed_scale_dec_pressed_(false)
  {
    // --- パラメータ宣言＆取得 ---
    // [変更] ボタン割り当て
    declare_parameter<int>("joy_enable_button", 4);    // 手動操縦有効化 (L1)
    declare_parameter<int>("steer_gain_button", 5);    // ステア高ゲイン (R1)
    declare_parameter<int>("autonomy_button",   3);    // 自動操縦モード (Yボタンなど)
    declare_parameter<int>("start_button",      9);
    declare_parameter<int>("stop_button",       8);

    // [変更] 軸の割り当て
    declare_parameter<int>("speed_axis", 1); // 左スティック上下
    declare_parameter<int>("steer_axis", 3); // 右スティック左右

    // [変更] スケール関連
    declare_parameter<double>("speed_scale", 1.0);
    declare_parameter<double>("base_steer_scale", 0.5);      // 通常時のステアスケール
    declare_parameter<double>("high_gain_steer_scale", 1.0); // 高ゲイン時のステアスケール
    declare_parameter<double>("scale_increment", 0.1);       // スケール変更のステップ幅

    declare_parameter<double>("timer_hz", 40.0);

    // パラメータ取得
    get_parameter("joy_enable_button", joy_enable_button_);
    get_parameter("steer_gain_button", steer_gain_button_);
    get_parameter("autonomy_button", autonomy_button_);
    get_parameter("start_button", start_button_);
    get_parameter("stop_button", stop_button_);
    get_parameter("speed_axis", speed_axis_);
    get_parameter("steer_axis", steer_axis_);
    get_parameter("speed_scale", speed_scale_);
    get_parameter("base_steer_scale", base_steer_scale_);
    get_parameter("high_gain_steer_scale", high_gain_steer_scale_);
    get_parameter("scale_increment", scale_increment_);
    get_parameter("timer_hz", timer_hz_);

    // 最後の自律走行コマンドを初期化
    last_autonomy_msg_.speed = 0.0;
    last_autonomy_msg_.steering_angle = 0.0;

    // --- サブスクライバ／パブリッシャ設定 ---
    joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10, std::bind(&AdvancedJoyManagerNode::joy_callback, this, _1));
    ack_sub_ = create_subscription<ackermann_msgs::msg::AckermannDrive>(
      "/ackermann_cmd", 10, std::bind(&AdvancedJoyManagerNode::ack_callback, this, _1));

    drive_pub_   = create_publisher<ackermann_msgs::msg::AckermannDrive>("/cmd_drive", 10);
    trigger_pub_ = create_publisher<std_msgs::msg::Bool>("/rosbag2_recorder/trigger", 10);
    
    // [削除] オフセット調整用のPublisherは不要

    // --- タイマー ---
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / timer_hz_),
      std::bind(&AdvancedJoyManagerNode::timer_callback, this));
    
    RCLCPP_INFO(get_logger(), "Advanced Joy Manager Node has been started.");
  }

private:
  // デバウンス付きボタン検出ヘルパー
  bool check_button_press(bool curr, bool &prev_flag) {
    if (curr && !prev_flag) {
      prev_flag = true;
      return true;
    } else if (!curr) {
      prev_flag = false;
    }
    return false;
  }

  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    // --- ボタン・軸の存在チェック ---
    // (可読性のために省略しますが、元のコードのようにサイズチェックを入れるのが安全です)

    // --- rosbag録画トリガー ---
    bool curr_start = (msg->buttons[start_button_] == 1);
    bool curr_stop  = (msg->buttons[stop_button_]  == 1);
    if (check_button_press(curr_start, prev_start_pressed_)) {
      std_msgs::msg::Bool b; b.data = true;
      trigger_pub_->publish(b);
      RCLCPP_INFO(get_logger(), "Published rosbag record START trigger.");
    }
    if (check_button_press(curr_stop, prev_stop_pressed_)) {
      std_msgs::msg::Bool b; b.data = false;
      trigger_pub_->publish(b);
      RCLCPP_INFO(get_logger(), "Published rosbag record STOP trigger.");
    }

    // --- [変更] モード・ゲインボタンの状態取得 ---
    joy_enable_pressed_ = (msg->buttons[joy_enable_button_] == 1);
    steer_gain_pressed_ = (msg->buttons[steer_gain_button_] == 1);
    autonomy_pressed_   = (msg->buttons[autonomy_button_]   == 1);

    // --- [変更] 手動操縦時のスティック入力取得 ---
    if (joy_enable_pressed_) {
      raw_speed_ = msg->axes[speed_axis_];
      raw_steer_ = msg->axes[steer_axis_];
    } else {
      raw_speed_ = 0.0;
      raw_steer_ = 0.0;
    }

    // --- [変更] D-padでのスケール動的調整 ---
    double dpad_lr = msg->axes[6]; // D-pad Left/Right
    double dpad_ud = msg->axes[7]; // D-pad Up/Down

    // D-pad 右: base_steer_scale を増加
    if (check_button_press(dpad_lr > 0.5, prev_steer_scale_inc_pressed_)) {
      base_steer_scale_ += scale_increment_;
      RCLCPP_INFO(get_logger(), "Base Steer Scale set to: %.2f", base_steer_scale_);
    }
    // D-pad 左: base_steer_scale を減少
    if (check_button_press(dpad_lr < -0.5, prev_steer_scale_dec_pressed_)) {
      base_steer_scale_ = std::max(0.0, base_steer_scale_ - scale_increment_);
      RCLCPP_INFO(get_logger(), "Base Steer Scale set to: %.2f", base_steer_scale_);
    }
    // D-pad 上: speed_scale を増加
    if (check_button_press(dpad_ud > 0.5, prev_speed_scale_inc_pressed_)) {
      speed_scale_ += scale_increment_;
      RCLCPP_INFO(get_logger(), "Speed Scale set to: %.2f", speed_scale_);
    }
    // D-pad 下: speed_scale を減少
    if (check_button_press(dpad_ud < -0.5, prev_speed_scale_dec_pressed_)) {
      speed_scale_ = std::max(0.0, speed_scale_ - scale_increment_);
      RCLCPP_INFO(get_logger(), "Speed Scale set to: %.2f", speed_scale_);
    }
  }

  void ack_callback(const ackermann_msgs::msg::AckermannDrive::SharedPtr msg)
  {
    last_autonomy_msg_ = *msg;
  }

  void timer_callback()
  {
    ackermann_msgs::msg::AckermannDrive out;

    // [変更] モードに応じたコマンドを生成
    if (autonomy_pressed_) {
      // 自動操縦モード
      out = last_autonomy_msg_;
    } else if (joy_enable_pressed_) {
      // 手動操縦モード
      double current_steer_scale = steer_gain_pressed_ ? high_gain_steer_scale_ : base_steer_scale_;
      
      out.speed          = raw_speed_ * speed_scale_;
      out.steering_angle = raw_steer_ * current_steer_scale;
    } else {
      // アイドルモード
      out.speed          = 0.0;
      out.steering_angle = 0.0;
    }
    drive_pub_->publish(out);
  }

  // --- メンバ変数 ---
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr           joy_sub_;
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDrive>::SharedPtr ack_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDrive>::SharedPtr drive_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr                trigger_pub_;
  rclcpp::TimerBase::SharedPtr                                     timer_;

  // パラメータ
  int joy_enable_button_, steer_gain_button_, autonomy_button_;
  int start_button_, stop_button_;
  int speed_axis_, steer_axis_;
  double speed_scale_, base_steer_scale_, high_gain_steer_scale_, scale_increment_;
  double timer_hz_;

  // 状態
  bool joy_enable_pressed_, steer_gain_pressed_, autonomy_pressed_;
  double raw_speed_, raw_steer_;
  ackermann_msgs::msg::AckermannDrive last_autonomy_msg_;

  // 連射防止フラグ
  bool prev_start_pressed_, prev_stop_pressed_;
  bool prev_steer_scale_inc_pressed_, prev_steer_scale_dec_pressed_;
  bool prev_speed_scale_inc_pressed_, prev_speed_scale_dec_pressed_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AdvancedJoyManagerNode>());
  rclcpp::shutdown();
  return 0;
}