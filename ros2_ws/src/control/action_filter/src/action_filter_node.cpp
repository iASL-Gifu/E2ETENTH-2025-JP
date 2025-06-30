#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive.hpp>
#include <deque>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

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
    this->declare_parameter<double>("speed_scale_ratio", 1.0); // 速度のスケール比を追加
    this->declare_parameter<double>("steer_scale_ratio", 1.0); // 操舵角のスケール比を追加
    
    // パラメータの取得
    this->get_parameter("filter_type", filter_type_);
    this->get_parameter("window_size", window_size_);
    this->get_parameter("use_scale_filter", use_scale_filter_);
    this->get_parameter("speed_scale_ratio", speed_scale_ratio_); // パラメータを取得
    this->get_parameter("steer_scale_ratio", steer_scale_ratio_); // パラメータを取得

    // パラメータ値のバリデーションと表示
    if (filter_type_ != "average" && filter_type_ != "median" && filter_type_ != "none") {
      RCLCPP_ERROR(this->get_logger(), "Invalid filter_type: %s. Using 'none'.", filter_type_.c_str());
      filter_type_ = "none";
    }
    if (window_size_ <= 0) {
      RCLCPP_ERROR(this->get_logger(), "window_size must be positive. Setting to 1.");
      window_size_ = 1;
    }

    RCLCPP_INFO(this->get_logger(), "--- Ackermann Filter Node Settings ---");
    RCLCPP_INFO(this->get_logger(), "Input topic: %s", INPUT_TOPIC);
    RCLCPP_INFO(this->get_logger(), "Output topic: %s", OUTPUT_TOPIC);
    RCLCPP_INFO(this->get_logger(), "Filter type: %s", filter_type_.c_str());
    if (filter_type_ != "none") {
      RCLCPP_INFO(this->get_logger(), "Window size: %d", window_size_);
    }
    RCLCPP_INFO(this->get_logger(), "Use scale/clip filter: %s", use_scale_filter_ ? "true" : "false");
    if (use_scale_filter_){
      RCLCPP_INFO(this->get_logger(), "Speed scale ratio: %.2f", speed_scale_ratio_);
      RCLCPP_INFO(this->get_logger(), "Steer scale ratio: %.2f", steer_scale_ratio_);
    }
    RCLCPP_INFO(this->get_logger(), "------------------------------------");

    // PublisherとSubscriberの初期化 (トピック名を直接指定)
    publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDrive>(OUTPUT_TOPIC, 10);
    subscription_ = this->create_subscription<ackermann_msgs::msg::AckermannDrive>(
      INPUT_TOPIC, 10, std::bind(&AckermannFilterNode::topic_callback, this, std::placeholders::_1));
  }

private:
  // 受信したメッセージを処理するコールバック関数
  void topic_callback(const ackermann_msgs::msg::AckermannDrive::SharedPtr msg)
  {
    // データをバッファに追加
    speed_buffer_.push_back(msg->speed);
    steering_angle_buffer_.push_back(msg->steering_angle);

    // バッファサイズがウィンドウサイズを超えたら古いデータを削除
    if (speed_buffer_.size() > static_cast<size_t>(window_size_)) {
      speed_buffer_.pop_front();
      steering_angle_buffer_.pop_front();
    }

    auto filtered_msg = ackermann_msgs::msg::AckermannDrive();
    filtered_msg.stamp = this->get_clock()->now(); // タイムスタンプを更新

    // フィルタを適用
    if (filter_type_ == "average") {
      apply_average_filter(filtered_msg);
    } else if (filter_type_ == "median") {
      apply_median_filter(filtered_msg);
    } else { // "none" or invalid
      // フィルタをかけない場合は最新の値をそのまま使用
      filtered_msg.speed = msg->speed;
      filtered_msg.steering_angle = msg->steering_angle;
    }
    
    // スケール/クリップフィルタを適用
    if (use_scale_filter_) {
      apply_scale_filter(filtered_msg);
    }

    // フィルタリング後のメッセージを発行
    publisher_->publish(filtered_msg);
  }

  // 移動平均フィルタ
  void apply_average_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    if (speed_buffer_.empty()) return;
    
    double speed_sum = std::accumulate(speed_buffer_.begin(), speed_buffer_.end(), 0.0);
    msg.speed = speed_sum / speed_buffer_.size();

    double steer_sum = std::accumulate(steering_angle_buffer_.begin(), steering_angle_buffer_.end(), 0.0);
    msg.steering_angle = steer_sum / steering_angle_buffer_.size();
  }

  // 移動中央値フィルタ
  void apply_median_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    if (speed_buffer_.empty()) return;

    msg.speed = calculate_median(speed_buffer_);
    msg.steering_angle = calculate_median(steering_angle_buffer_);
  }
  
  // dequeから中央値を計算するヘルパー関数
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

  // スケール/クリップフィルタ 
  void apply_scale_filter(ackermann_msgs::msg::AckermannDrive &msg)
  {
    // パラメータで指定された倍率を掛ける
    msg.speed *= speed_scale_ratio_;
    msg.steering_angle *= steer_scale_ratio_;
    
    // 指定された範囲に値をクリップ(clamp)する
    msg.speed = std::clamp(msg.speed, 0.0, 1.0);
    msg.steering_angle = std::clamp(msg.steering_angle, -1.0, 1.0);
  }

  // メンバ変数
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDrive>::SharedPtr subscription_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDrive>::SharedPtr publisher_;
  
  std::string filter_type_;
  int window_size_;
  bool use_scale_filter_;
  double speed_scale_ratio_; // 速度のスケール比を保持するメンバ変数
  double steer_scale_ratio_; // 操舵角のスケール比を保持するメンバ変数

  std::deque<double> speed_buffer_;
  std::deque<double> steering_angle_buffer_;
};

// main関数
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AckermannFilterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}