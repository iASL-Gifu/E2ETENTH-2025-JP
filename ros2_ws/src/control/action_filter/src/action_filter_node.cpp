#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive.hpp>
#include <deque>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cmath>
#include "rcl_interfaces/msg/set_parameters_result.hpp"

// 固定するトピック名
const char* INPUT_TOPIC = "/cmd_drive";
const char* OUTPUT_TOPIC = "/cmd_drive_filtered";

// Nodeクラスを定義
class AckermannFilterNode : public rclcpp::Node
{
public:
  AckermannFilterNode()
  : Node("ackermann_filter_node")
  {
    // === パラメータの宣言 (ネスト構造に対応) ===
    this->declare_parameter<std::string>("filter_type", "none");
    this->declare_parameter<int>("window_size", 5);
    this->declare_parameter<bool>("use_scale_filter", true);
    this->declare_parameter<std::string>("scale_filter_type", "normal");

    // normalモード用のパラメータ
    this->declare_parameter<double>("normal.speed_scale_ratio", 1.0);
    this->declare_parameter<double>("normal.steer_scale_ratio", 1.0);

    // advanceモード用のパラメータ
    this->declare_parameter<double>("advance.straight_steer_threshold", 0.1);
    this->declare_parameter<double>("advance.straight_speed_scale_ratio", 1.0);
    this->declare_parameter<double>("advance.cornering_speed_scale_ratio", 0.5);
    this->declare_parameter<double>("advance.steer_scale_ratio", 1.0);
    
    // === パラメータの初期値を取得 (分離したメンバ変数へ) ===
    this->get_parameter("filter_type", filter_type_);
    this->get_parameter("window_size", window_size_);
    this->get_parameter("use_scale_filter", use_scale_filter_);
    this->get_parameter("scale_filter_type", scale_filter_type_);
    
    this->get_parameter("normal.speed_scale_ratio", normal_speed_scale_ratio_);
    this->get_parameter("normal.steer_scale_ratio", normal_steer_scale_ratio_);

    this->get_parameter("advance.straight_steer_threshold", advance_straight_steer_threshold_);
    this->get_parameter("advance.straight_speed_scale_ratio", advance_straight_speed_scale_ratio_);
    this->get_parameter("advance.cornering_speed_scale_ratio", advance_cornering_speed_scale_ratio_);
    this->get_parameter("advance.steer_scale_ratio", advance_steer_scale_ratio_);

    // 起動時のパラメータ情報を表示
    print_parameters();

    // 動的パラメータ変更のためのコールバックを登録
    parameters_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&AckermannFilterNode::parameters_callback, this, std::placeholders::_1));

    // PublisherとSubscriberの初期化
    publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDrive>(OUTPUT_TOPIC, 10);
    subscription_ = this->create_subscription<ackermann_msgs::msg::AckermannDrive>(
      INPUT_TOPIC, 10, std::bind(&AckermannFilterNode::topic_callback, this, std::placeholders::_1));
  }

private:
  void topic_callback(const ackermann_msgs::msg::AckermannDrive::SharedPtr msg)
  {
    speed_buffer_.push_back(msg->speed);
    steering_angle_buffer_.push_back(msg->steering_angle);

    while (speed_buffer_.size() > static_cast<size_t>(window_size_)) {
      speed_buffer_.pop_front();
      steering_angle_buffer_.pop_front();
    }

    auto filtered_msg = ackermann_msgs::msg::AckermannDrive();
    
    if (filter_type_ == "average") {
      apply_average_filter(filtered_msg);
    } else if (filter_type_ == "median") {
      apply_median_filter(filtered_msg);
    } else {
      if (!speed_buffer_.empty()) {
        filtered_msg.speed = speed_buffer_.back();
        filtered_msg.steering_angle = steering_angle_buffer_.back();
      } else {
        filtered_msg.speed = msg->speed;
        filtered_msg.steering_angle = msg->steering_angle;
      }
    }
    
    if (use_scale_filter_) {
      if (scale_filter_type_ == "advance") {
          apply_advanced_scale_filter(filtered_msg);
      } else {
          apply_normal_scale_filter(filtered_msg);
      }
    }

    publisher_->publish(filtered_msg);
  }
  
  rcl_interfaces::msg::SetParametersResult parameters_callback(
    const std::vector<rclcpp::Parameter> &parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";

    for (const auto &param : parameters) {
      const std::string param_name = param.get_name();
      RCLCPP_INFO(this->get_logger(), "Parameter '%s' changed.", param_name.c_str());

      // === パラメータ名チェックをネスト構造に対応 ===
      if (param_name == "filter_type") {
        filter_type_ = param.as_string();
      } else if (param_name == "window_size") {
        window_size_ = param.as_int();
      } else if (param_name == "use_scale_filter") {
        use_scale_filter_ = param.as_bool();
      } else if (param_name == "scale_filter_type") {
        scale_filter_type_ = param.as_string();
      } else if (param_name == "normal.speed_scale_ratio") {
        normal_speed_scale_ratio_ = param.as_double();
      } else if (param_name == "normal.steer_scale_ratio") {
        normal_steer_scale_ratio_ = param.as_double();
      } else if (param_name == "advance.straight_steer_threshold") {
        advance_straight_steer_threshold_ = param.as_double();
      } else if (param_name == "advance.straight_speed_scale_ratio") {
        advance_straight_speed_scale_ratio_ = param.as_double();
      } else if (param_name == "advance.cornering_speed_scale_ratio") {
        advance_cornering_speed_scale_ratio_ = param.as_double();
      } else if (param_name == "advance.steer_scale_ratio") {
        advance_steer_scale_ratio_ = param.as_double();
      }
    }
    
    if (result.successful) {
        RCLCPP_INFO(this->get_logger(), "New parameters have been applied.");
        print_parameters();
    }

    return result;
  }
  
  void print_parameters() {
    RCLCPP_INFO(this->get_logger(), "--- Ackermann Filter Node Settings ---");
    RCLCPP_INFO(this->get_logger(), "Filter type: %s", filter_type_.c_str());
    if (filter_type_ != "none") {
      RCLCPP_INFO(this->get_logger(), "Window size: %d", window_size_);
    }
    RCLCPP_INFO(this->get_logger(), "Use scale filter: %s", use_scale_filter_ ? "true" : "false");

    if (use_scale_filter_){
      RCLCPP_INFO(this->get_logger(), "Scale filter type: %s", scale_filter_type_.c_str());
      // === ログ表示を新しい変数に対応 ===
      if (scale_filter_type_ == "advance") {
          RCLCPP_INFO(this->get_logger(), "  [advance] Straight steer threshold: %.2f rad", advance_straight_steer_threshold_);
          RCLCPP_INFO(this->get_logger(), "  [advance] Straight speed scale ratio: %.2f", advance_straight_speed_scale_ratio_);
          RCLCPP_INFO(this->get_logger(), "  [advance] Cornering speed scale ratio: %.2f", advance_cornering_speed_scale_ratio_);
          RCLCPP_INFO(this->get_logger(), "  [advance] Steer scale ratio: %.2f", advance_steer_scale_ratio_);
      } else { // normal
          RCLCPP_INFO(this->get_logger(), "  [normal] Speed scale ratio: %.2f", normal_speed_scale_ratio_);
          RCLCPP_INFO(this->get_logger(), "  [normal] Steer scale ratio: %.2f", normal_steer_scale_ratio_);
      }
    }
    RCLCPP_INFO(this->get_logger(), "------------------------------------");
  }

  void apply_normal_scale_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    msg.speed *= normal_speed_scale_ratio_;
    msg.steering_angle *= normal_steer_scale_ratio_;
    
    msg.speed = std::max(0.0f, std::min(msg.speed, 1.0f));
    msg.steering_angle = std::max(-1.0f, std::min(msg.steering_angle, 1.0f));
  }

  void apply_advanced_scale_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    if (std::fabs(msg.steering_angle) < advance_straight_steer_threshold_) {
        msg.speed *= advance_straight_speed_scale_ratio_;
    } else {
        msg.speed *= advance_cornering_speed_scale_ratio_;
    }
    msg.steering_angle *= advance_steer_scale_ratio_;

    // ★ 修正点: 0.0, 1.0, -1.0 を float型リテラル (f付き) に変更
    msg.speed = std::max(0.0f, std::min(msg.speed, 1.0f));
    msg.steering_angle = std::max(-1.0f, std::min(msg.steering_angle, 1.0f));
  }

  void apply_average_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    if (speed_buffer_.empty()) return;
    double speed_sum = std::accumulate(speed_buffer_.begin(), speed_buffer_.end(), 0.0);
    msg.speed = speed_sum / speed_buffer_.size();
    double steer_sum = std::accumulate(steering_angle_buffer_.begin(), steering_angle_buffer_.end(), 0.0);
    msg.steering_angle = steer_sum / steering_angle_buffer_.size();
  }
  void apply_median_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    if (speed_buffer_.empty()) return;
    msg.speed = calculate_median(speed_buffer_);
    msg.steering_angle = calculate_median(steering_angle_buffer_);
  }
  double calculate_median(const std::deque<double>& data)
  {
    if (data.empty()) return 0.0;
    std::vector<double> sorted_data(data.begin(), data.end());
    size_t n = sorted_data.size();
    std::sort(sorted_data.begin(), sorted_data.end());
    if (n % 2 == 0) {
        return (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0;
    } else {
        return sorted_data[n / 2];
    }
  }
  // --- ここまで変更のないヘルパー関数 ---

  rclcpp::Subscription<ackermann_msgs::msg::AckermannDrive>::SharedPtr subscription_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDrive>::SharedPtr publisher_;
  OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;
  
  // === メンバ変数をモードごとに分離 ===
  // 一般設定
  std::string filter_type_;
  int window_size_;
  bool use_scale_filter_;
  std::string scale_filter_type_;

  // normalモード用パラメータ
  double normal_speed_scale_ratio_;
  double normal_steer_scale_ratio_;
  
  // advanceモード用パラメータ
  double advance_straight_steer_threshold_;
  double advance_straight_speed_scale_ratio_;
  double advance_cornering_speed_scale_ratio_;
  double advance_steer_scale_ratio_;

  // データバッファ
  std::deque<double> speed_buffer_;
  std::deque<double> steering_angle_buffer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AckermannFilterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}