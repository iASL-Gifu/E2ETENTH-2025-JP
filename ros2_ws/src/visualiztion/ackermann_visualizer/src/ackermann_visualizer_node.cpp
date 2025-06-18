#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive.hpp>
#include <std_msgs/msg/float32.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include <opencv2/opencv.hpp>

using AckermannDrive = ackermann_msgs::msg::AckermannDrive;
using Float32 = std_msgs::msg::Float32;
using Image = sensor_msgs::msg::Image;

const int IMAGE_WIDTH = 600;
const int IMAGE_HEIGHT = 400;

class AckermannVisualizer : public rclcpp::Node
{
public:
    AckermannVisualizer() : Node("ackermann_visualizer_node")
    {
        // --- パラメータの宣言と取得 ---
        this->declare_parameter("invert_x_axis", false);
        invert_x_axis_ = this->get_parameter("invert_x_axis").as_bool();

        this->declare_parameter("invert_y_axis", false);
        invert_y_axis_ = this->get_parameter("invert_y_axis").as_bool();

        this->declare_parameter("center_zero_x", true);
        center_zero_x_ = this->get_parameter("center_zero_x").as_bool();

        this->declare_parameter("center_zero_y", false);
        center_zero_y_ = this->get_parameter("center_zero_y").as_bool();

        this->declare_parameter("x_axis_min", -1.0);
        x_axis_min_ = this->get_parameter("x_axis_min").as_double();
        
        this->declare_parameter("x_axis_max", 1.0);
        x_axis_max_ = this->get_parameter("x_axis_max").as_double();

        this->declare_parameter("y_axis_min", -1.0);
        y_axis_min_ = this->get_parameter("y_axis_min").as_double();

        this->declare_parameter("y_axis_max", 1.0);
        y_axis_max_ = this->get_parameter("y_axis_max").as_double();

        // **新しいパラメータ: 履歴点の最大数**
        this->declare_parameter("max_history_points", 500); // デフォルト500点
        max_history_points_ = this->get_parameter("max_history_points").as_int();


        // QoS設定 (以前と同じ)
        rclcpp::QoS qos_profile_rclcpp(10);
        rmw_qos_profile_t qos_profile_rmw = qos_profile_rclcpp.get_rmw_qos_profile();

        ackermann_sub_ = this->create_subscription<AckermannDrive>(
            "/jetracer/cmd_drive", qos_profile_rclcpp,
            std::bind(&AckermannVisualizer::ackermannCallback, this, std::placeholders::_1));

        speed_pub_ = this->create_publisher<Float32>("visualize/speed", qos_profile_rclcpp);
        steer_pub_ = this->create_publisher<Float32>("visualize/steer", qos_profile_rclcpp);

        image_pub_ = image_transport::create_publisher(this, "/visualize/ackermann_plot_image", qos_profile_rmw);

        RCLCPP_INFO(this->get_logger(), "Ackermann Visualizer Node Started with parameters:");
        RCLCPP_INFO(this->get_logger(), "  invert_x_axis: %s", invert_x_axis_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  invert_y_axis: %s", invert_y_axis_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  center_zero_x: %s", center_zero_x_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  center_zero_y: %s", center_zero_y_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  x_axis_min: %.2f", x_axis_min_);
        RCLCPP_INFO(this->get_logger(), "  x_axis_max: %.2f", x_axis_max_);
        RCLCPP_INFO(this->get_logger(), "  y_axis_min: %.2f", y_axis_min_);
        RCLCPP_INFO(this->get_logger(), "  y_axis_max: %.2f", y_axis_max_);
        RCLCPP_INFO(this->get_logger(), "  max_history_points: %d", max_history_points_); // <-- 追加
        RCLCPP_INFO(this->get_logger(), "Subscribing to /jetracer/cmd_drive");
        RCLCPP_INFO(this->get_logger(), "Publishing to /visualize/speed, /visualize/steer, /visualize/ackermann_plot_image");
    }

private:
    rclcpp::Subscription<AckermannDrive>::SharedPtr ackermann_sub_;
    rclcpp::Publisher<Float32>::SharedPtr speed_pub_;
    rclcpp::Publisher<Float32>::SharedPtr steer_pub_;
    image_transport::Publisher image_pub_;

    // パラメータを保存するメンバ変数 (履歴点の最大数も追加)
    bool invert_x_axis_;
    bool invert_y_axis_;
    bool center_zero_x_;
    bool center_zero_y_;
    double x_axis_min_;
    double x_axis_max_;
    double y_axis_min_;
    double y_axis_max_;
    int max_history_points_; // <-- 追加

    std::vector<cv::Point2f> historical_points_;
    // const size_t MAX_HISTORY_POINTS = 500; // <-- 削除またはコメントアウト。メンバ変数を使う

    void ackermannCallback(const AckermannDrive::SharedPtr msg)
    {
        float current_speed = msg->speed;
        float current_steer = msg->steering_angle;

        Float32 speed_msg;
        speed_msg.data = current_speed;
        speed_pub_->publish(speed_msg);

        Float32 steer_msg;
        steer_msg.data = current_steer;
        steer_pub_->publish(steer_msg);

        // RCLCPP_INFO(this->get_logger(), "Received - Speed: %.2f m/s, Steer: %.2f rad (%.2f deg)",
        //          current_speed, current_steer, current_steer * 180.0 / M_PI);

        publishPlotImage(current_speed, current_steer);
    }

    void publishPlotImage(float speed, float steer)
    {
        if (image_pub_.getNumSubscribers() == 0) {
            return;
        }

        cv::Mat image = cv::Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);

        float min_speed_plot = y_axis_min_;
        float max_speed_plot = y_axis_max_;
        float speed_range_plot = max_speed_plot - min_speed_plot;

        float min_steer_plot = x_axis_min_;
        float max_steer_plot = x_axis_max_;
        float steer_range_plot = max_steer_plot - min_steer_plot;


        // Y軸 (速度) の描画
        int y_axis_x_pos = 50;
        cv::line(image, cv::Point(y_axis_x_pos, 0), cv::Point(y_axis_x_pos, IMAGE_HEIGHT), cv::Scalar(255, 255, 255), 1);
        cv::putText(image, "Speed (m/s)", cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        float y_tick_interval = (speed_range_plot > 5.0) ? 1.0 : ((speed_range_plot > 2.0) ? 0.5 : 0.2);
        for (float s = min_speed_plot; s <= max_speed_plot + y_tick_interval/2; s += y_tick_interval) {
            int y = IMAGE_HEIGHT - (int)(((s - min_speed_plot) / speed_range_plot) * IMAGE_HEIGHT);
            if (invert_y_axis_) y = IMAGE_HEIGHT - y;
            
            if (center_zero_y_ && std::abs(s) < y_tick_interval / 4) { // 浮動小数点比較の安全な方法
                cv::line(image, cv::Point(0, y), cv::Point(IMAGE_WIDTH, y), cv::Scalar(100, 100, 100), 1); // 灰色
            }
            cv::line(image, cv::Point(y_axis_x_pos - 5, y), cv::Point(y_axis_x_pos + 5, y), cv::Scalar(255, 255, 255), 1);
            std::string label = std::to_string(s);
            label = label.substr(0, label.find(".") + 3);
            cv::putText(image, label, cv::Point(y_axis_x_pos + 15, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }

        // X軸 (操舵角) の描画
        int x_axis_y_pos;
        if (center_zero_y_) {
            x_axis_y_pos = IMAGE_HEIGHT - (int)(((0.0f - min_speed_plot) / speed_range_plot) * IMAGE_HEIGHT);
            if (invert_y_axis_) x_axis_y_pos = IMAGE_HEIGHT - x_axis_y_pos;
        } else {
            x_axis_y_pos = IMAGE_HEIGHT - 50;
        }
        cv::line(image, cv::Point(0, x_axis_y_pos), cv::Point(IMAGE_WIDTH, x_axis_y_pos), cv::Scalar(255, 255, 255), 1);
        cv::putText(image, "Steering Angle (rad)", cv::Point(IMAGE_WIDTH / 2 - 80, IMAGE_HEIGHT - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        float x_tick_interval = (steer_range_plot > 2.0) ? 0.5 : 0.25;
        for (float st = min_steer_plot; st <= max_steer_plot + x_tick_interval/2; st += x_tick_interval) {
            int x = (int)(((st - min_steer_plot) / steer_range_plot) * IMAGE_WIDTH);
            if (invert_x_axis_) x = IMAGE_WIDTH - x;

            if (center_zero_x_ && std::abs(st) < x_tick_interval / 4) { // 浮動小数点比較の安全な方法
                cv::line(image, cv::Point(x, 0), cv::Point(x, IMAGE_HEIGHT), cv::Scalar(100, 100, 100), 1); // 灰色
            }
            cv::line(image, cv::Point(x - 5, x_axis_y_pos), cv::Point(x + 5, x_axis_y_pos), cv::Scalar(255, 255, 255), 1);
            std::string label = std::to_string(st);
            label = label.substr(0, label.find(".") + 3);
            cv::putText(image, label, cv::Point(x - 15, x_axis_y_pos + 25), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }


        // --- 現在の速度と操舵角を画像座標にマッピング ---
        int plot_x, plot_y;

        float normalized_steer = (steer - min_steer_plot) / steer_range_plot;
        plot_x = static_cast<int>(normalized_steer * IMAGE_WIDTH);
        if (invert_x_axis_) plot_x = IMAGE_WIDTH - plot_x;

        float normalized_speed = (speed - min_speed_plot) / speed_range_plot;
        plot_y = IMAGE_HEIGHT - static_cast<int>(normalized_speed * IMAGE_HEIGHT);
        if (invert_y_axis_) plot_y = IMAGE_HEIGHT - plot_y;


        // プロット領域からはみ出さないようにクリップ
        plot_x = std::max(0, std::min(IMAGE_WIDTH - 1, plot_x));
        plot_y = std::max(0, std::min(IMAGE_HEIGHT - 1, plot_y));

        // 過去の点を保存し、最新の点に色を付ける
        // MAX_HISTORY_POINTS の代わりに max_history_points_ メンバ変数を使用
        historical_points_.push_back(cv::Point2f(plot_x, plot_y));
        if (historical_points_.size() > static_cast<size_t>(max_history_points_)) { // size_t にキャスト
            historical_points_.erase(historical_points_.begin());
        }

        // 過去の点を描画 (以前と同じ)
        for (size_t i = 0; i < historical_points_.size(); ++i) {
            float alpha = (float)(i + 1) / historical_points_.size();
            
            cv::Scalar point_color;
            float clamped_speed = std::max(min_speed_plot, std::min(max_speed_plot, speed));
            float color_ratio = (clamped_speed - min_speed_plot) / speed_range_plot;
            
            point_color = cv::Scalar(255 * (1.0 - color_ratio), 0, 255 * color_ratio);

            point_color[0] *= alpha;
            point_color[1] *= alpha;
            point_color[2] *= alpha;

            cv::circle(image, historical_points_[i], 2, point_color, cv::FILLED);
        }

        cv::circle(image, cv::Point(plot_x, plot_y), 4, cv::Scalar(0, 0, 255), cv::FILLED);

        std::string speed_str = "Speed: " + std::to_string(speed).substr(0, std::to_string(speed).find(".") + 3) + " m/s";
        std::string steer_str = "Steer: " + std::to_string(steer).substr(0, std::to_string(steer).find(".") + 3) + " rad";
        cv::putText(image, speed_str, cv::Point(IMAGE_WIDTH - 150, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::putText(image, steer_str, cv::Point(IMAGE_WIDTH - 150, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);


        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        header.frame_id = "ackermann_plot_frame";

        sensor_msgs::msg::Image::SharedPtr ros_image_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        image_pub_.publish(*ros_image_msg);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AckermannVisualizer>());
    rclcpp::shutdown();
    return 0;
}