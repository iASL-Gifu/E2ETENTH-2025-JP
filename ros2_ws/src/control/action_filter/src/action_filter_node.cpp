#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive.hpp>
#include <deque>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm> // std::sort, std::min, std::max のために必要
#include <functional>
#include <cmath> 

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
    // パラメータの宣言とデフォルト値の設定
    this->declare_parameter<std::string>("filter_type", "none");
    this->declare_parameter<int>("window_size", 5);
    this->declare_parameter<bool>("use_scale_filter", true);

    this->declare_parameter<std::string>("scale_filter_type", "normal"); // スケールフィルタの種類を追加 (normal or advance)
    this->declare_parameter<double>("speed_scale_ratio", 1.0);
    this->declare_parameter<double>("steer_scale_ratio", 1.0);
    // 'advance'モード用のパラメータを追加
    this->declare_parameter<double>("straight_steer_threshold", 0.1); // 直進と判断する操舵角の閾値(rad)
    this->declare_parameter<double>("cornering_speed_scale_ratio", 0.5); // カーブ時の速度スケール比
    
    // パラメータの取得
    this->get_parameter("filter_type", filter_type_);
    this->get_parameter("window_size", window_size_);
    this->get_parameter("use_scale_filter", use_scale_filter_);
    this->get_parameter("scale_filter_type", scale_filter_type_);
    this->get_parameter("speed_scale_ratio", speed_scale_ratio_);
    this->get_parameter("steer_scale_ratio", steer_scale_ratio_);
    this->get_parameter("straight_steer_threshold", straight_steer_threshold_);
    this->get_parameter("cornering_speed_scale_ratio", cornering_speed_scale_ratio_);

    // パラメータ値のバリデーションと表示
    if (filter_type_ != "average" && filter_type_ != "median" && filter_type_ != "none") {
      RCLCPP_ERROR(this->get_logger(), "Invalid filter_type: %s. Using 'none'.", filter_type_.c_str());
      filter_type_ = "none";
    }
    if (window_size_ <= 0) {
      RCLCPP_ERROR(this->get_logger(), "window_size must be positive. Setting to 1.");
      window_size_ = 1;
    }

    if (scale_filter_type_ != "normal" && scale_filter_type_ != "advance") {
      RCLCPP_ERROR(this->get_logger(), "Invalid scale_filter_type: %s. Using 'normal'.", scale_filter_type_.c_str());
      scale_filter_type_ = "normal";
    }

    RCLCPP_INFO(this->get_logger(), "--- Ackermann Filter Node Settings ---");
    RCLCPP_INFO(this->get_logger(), "Input topic: %s", INPUT_TOPIC);
    RCLCPP_INFO(this->get_logger(), "Output topic: %s", OUTPUT_TOPIC);
    RCLCPP_INFO(this->get_logger(), "Filter type: %s", filter_type_.c_str());
    if (filter_type_ != "none") {
      RCLCPP_INFO(this->get_logger(), "Window size: %d", window_size_);
    }
    RCLCPP_INFO(this->get_logger(), "Use scale filter: %s", use_scale_filter_ ? "true" : "false");
    if (use_scale_filter_){
      RCLCPP_INFO(this->get_logger(), "Scale filter type: %s", scale_filter_type_.c_str());
      if (scale_filter_type_ == "advance") {
          RCLCPP_INFO(this->get_logger(), "  Straight steer threshold: %.2f rad", straight_steer_threshold_);
          RCLCPP_INFO(this->get_logger(), "  Straight speed scale ratio: %.2f", speed_scale_ratio_);
          RCLCPP_INFO(this->get_logger(), "  Cornering speed scale ratio: %.2f", cornering_speed_scale_ratio_);
          RCLCPP_INFO(this->get_logger(), "  Steer scale ratio: %.2f", steer_scale_ratio_);
      } else { // normal
          RCLCPP_INFO(this->get_logger(), "  Speed scale ratio: %.2f", speed_scale_ratio_);
          RCLCPP_INFO(this->get_logger(), "  Steer scale ratio: %.2f", steer_scale_ratio_);
      }
    }
    RCLCPP_INFO(this->get_logger(), "------------------------------------");

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

    if (speed_buffer_.size() > static_cast<size_t>(window_size_)) {
      speed_buffer_.pop_front();
      steering_angle_buffer_.pop_front();
    }

    auto filtered_msg = ackermann_msgs::msg::AckermannDrive();
    

    if (filter_type_ == "average") {
      apply_average_filter(filtered_msg);
    } else if (filter_type_ == "median") {
      apply_median_filter(filtered_msg);
    } else {
      filtered_msg.speed = msg->speed;
      filtered_msg.steering_angle = msg->steering_angle;
    }
    
    if (use_scale_filter_) {
      // 選択されたスケールフィルタを適用
      if (scale_filter_type_ == "advance") {
          apply_advanced_scale_filter(filtered_msg);
      } else { // "normal"
          apply_normal_scale_filter(filtered_msg);
      }
    }

    publisher_->publish(filtered_msg);
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
    std::vector<double> sorted_data(data.begin(), data.end());
    size_t n = sorted_data.size();
    std::sort(sorted_data.begin(), sorted_data.end());
    if (n == 0) return 0.0;
    if (n % 2 == 0) {
        return (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0;
    } else {
        return sorted_data[n / 2];
    }
  }

  // 元のスケールフィルタを 'normal' バージョンとしてリネーム
  void apply_normal_scale_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    msg.speed *= speed_scale_ratio_;
    msg.steering_angle *= steer_scale_ratio_;
    
    msg.speed = std::max(0.0f, std::min(msg.speed, 1.0f));
    msg.steering_angle = std::max(-1.0f, std::min(msg.steering_angle, 1.0f));
  }

  // 新しい 'advance' スケールフィルタ
  void apply_advanced_scale_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    // 現在の（移動平均/中央値フィルタ適用後の）操舵角に基づいて状態を判断
    // 操舵角の絶対値が閾値より小さい場合、'直進状態'と判断
    if (std::fabs(msg.steering_angle) < straight_steer_threshold_) {
        // 直進状態では、パラメータで指定された通常の速度スケールを適用
        msg.speed *= speed_scale_ratio_;
    } else {
        // カーブ状態では、専用のカーブ時速度スケールを適用
        msg.speed *= cornering_speed_scale_ratio_;
    }

    // 操舵角のスケールは常に一定
    msg.steering_angle *= steer_scale_ratio_;

    msg.speed = std::max(0.0f, std::min(msg.speed, 1.0f));
    msg.steering_angle = std::max(-1.0f, std::min(msg.steering_angle, 1.0f));
  }
  
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDrive>::SharedPtr subscription_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDrive>::SharedPtr publisher_;
  
  std::string filter_type_;
  int window_size_;
  bool use_scale_filter_;
  std::string scale_filter_type_;
  double speed_scale_ratio_;
  double steer_scale_ratio_;
  double straight_steer_threshold_;
  double cornering_speed_scale_ratio_;
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