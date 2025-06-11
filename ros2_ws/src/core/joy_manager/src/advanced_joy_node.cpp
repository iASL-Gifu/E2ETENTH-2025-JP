#include <algorithm>
#include <chrono>
#include <memory>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "ackermann_msgs/msg/ackermann_drive.hpp"
#include "std_msgs/msg/bool.hpp" // [追加] std_msgs/msg/Bool を使うためにインクルード

using namespace std::chrono_literals;

class AdvancedJoyNode : public rclcpp::Node
{
public:
  AdvancedJoyNode()
  : Node("advanced_joy_node"),
    joy_active_(false),
    throttle_button_pressed_(false),
    steer_gain_active_(false),
    current_throttle_(0.0),
    raw_steer_(0.0),
    prev_start_pressed_(false), 
    prev_stop_pressed_(false)   
  {
    // --- パラメータ宣言＆取得 ---
    // ボタン・軸の割り当て
    declare_parameter<int>("joy_enable_button", 5);
    declare_parameter<int>("throttle_button", 0);
    declare_parameter<int>("steer_gain_button", 1);
    declare_parameter<int>("steer_axis", 0);
    declare_parameter<int>("start_button", 7); 
    declare_parameter<int>("stop_button", 6);  

    // スロットル（アクセル）関連パラメータ
    declare_parameter<double>("max_throttle", 2.0);
    declare_parameter<double>("throttle_increase_rate", 1.5);
    declare_parameter<double>("throttle_decrease_rate", 4.0);

    // ステアリング関連パラメータ
    declare_parameter<double>("base_steer_scale", 1.0);
    declare_parameter<double>("high_gain_steer_scale", 1.5);

    // システム関連パラメータ
    declare_parameter<double>("timer_hz", 50.0);

    // パラメータ取得
    get_parameter("joy_enable_button", joy_enable_button_);
    get_parameter("throttle_button", throttle_button_);
    get_parameter("steer_gain_button", steer_gain_button_);
    get_parameter("steer_axis", steer_axis_);
    get_parameter("start_button", start_button_); 
    get_parameter("stop_button", stop_button_);  
    get_parameter("max_throttle", max_throttle_);
    get_parameter("throttle_increase_rate", throttle_increase_rate_);
    get_parameter("throttle_decrease_rate", throttle_decrease_rate_);
    get_parameter("base_steer_scale", base_steer_scale_);
    get_parameter("high_gain_steer_scale", high_gain_steer_scale_);
    get_parameter("timer_hz", timer_hz_);

    // --- サブスクライバ／パブリッシャ設定 ---
    joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10, std::bind(&MarioKartJoyNode::joy_callback, this, _1));

    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDrive>("/cmd_drive", 10);
    // [追加] rosbag記録トリガー用のPublisher
    trigger_pub_ = create_publisher<std_msgs::msg::Bool>("/rosbag2_recorder/trigger", 10);

    // --- タイマー ---
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / timer_hz_),
      std::bind(&MarioKartJoyNode::timer_callback, this));

    RCLCPP_INFO(get_logger(), "Mario Kart Style Joy Node with Trigger has been started.");
  }

private:
  bool check_button_press(bool curr_state, bool &prev_state) {
    if (curr_state && !prev_state) {
      prev_state = true;
      return true;
    } else if (!curr_state) {
      prev_state = false;
    }
    return false;
  }

  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    // [変更] ボタンの数が足りない場合のチェックを更新
    if (msg->buttons.size() <= std::max({joy_enable_button_, throttle_button_, steer_gain_button_, start_button_, stop_button_})) {
        RCLCPP_WARN_ONCE(get_logger(), "Joystick has not enough buttons for all functions.");
        return;
    }
    if (msg->axes.size() <= steer_axis_) {
        RCLCPP_WARN_ONCE(get_logger(), "Joystick has not enough axes.");
        return;
    }

    bool curr_start = (msg->buttons[start_button_] == 1);
    bool curr_stop  = (msg->buttons[stop_button_]  == 1);

    if (check_button_press(curr_start, prev_start_pressed_)) {
      auto trigger_msg = std_msgs::msg::Bool();
      trigger_msg.data = true;
      trigger_pub_->publish(trigger_msg);
      RCLCPP_INFO(get_logger(), "Published rosbag record START trigger.");
    }
    if (check_button_press(curr_stop, prev_stop_pressed_)) {
      auto trigger_msg = std_msgs::msg::Bool();
      trigger_msg.data = false;
      trigger_pub_->publish(trigger_msg);
      RCLCPP_INFO(get_logger(), "Published rosbag record STOP trigger.");
    }

    // 1. 各ボタンの押下状態を更新 (ホールド操作)
    joy_active_ = (msg->buttons[joy_enable_button_] == 1);
    throttle_button_pressed_ = (msg->buttons[throttle_button_] == 1);
    steer_gain_active_ = (msg->buttons[steer_gain_button_] == 1);

    // 2. ステアリングの生データを更新
    if (joy_active_) {
        raw_steer_ = msg->axes[steer_axis_];
    } else {
        raw_steer_ = 0.0;
    }
  }

  void timer_callback()
  {
    // (この関数の中身は変更ありません)
    double dt = 1.0 / timer_hz_;

    if (joy_active_ && throttle_button_pressed_) {
      current_throttle_ += throttle_increase_rate_ * dt;
      current_throttle_ = std::min(current_throttle_, max_throttle_);
    } else {
      current_throttle_ -= throttle_decrease_rate_ * dt;
      current_throttle_ = std::max(current_throttle_, 0.0);
    }

    double current_steer_scale = steer_gain_active_ ? high_gain_steer_scale_ : base_steer_scale_;
    double final_steer = raw_steer_ * current_steer_scale;

    auto drive_msg = ackermann_msgs::msg::AckermannDrive();

    if (joy_active_) {
      drive_msg.speed          = current_throttle_;
      drive_msg.steering_angle = final_steer;
    } else {
      drive_msg.speed          = 0.0;
      drive_msg.steering_angle = 0.0;
    }
    drive_pub_->publish(drive_msg);
  }

  // --- メンバ変数 ---
  // サブスクライバ／パブリッシャ
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDrive>::SharedPtr drive_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr trigger_pub_; 
  rclcpp::TimerBase::SharedPtr timer_;

  // パラメータ
  int joy_enable_button_, throttle_button_, steer_gain_button_, steer_axis_;
  int start_button_, stop_button_; 
  double max_throttle_, throttle_increase_rate_, throttle_decrease_rate_;
  double base_steer_scale_, high_gain_steer_scale_;
  double timer_hz_;

  // 状態変数
  bool joy_active_;
  bool throttle_button_pressed_;
  bool steer_gain_active_;
  double current_throttle_;
  double raw_steer_;
  bool prev_start_pressed_, prev_stop_pressed_; 
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AdvancedJoyNode>());
  rclcpp::shutdown();
  return 0;
}