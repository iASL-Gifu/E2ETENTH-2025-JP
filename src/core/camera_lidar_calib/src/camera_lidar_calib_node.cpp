#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class SensorListener : public rclcpp::Node
{
public:
    SensorListener()
    : Node("sensor_listener")
    {
        // パラメータ宣言と読み込み
        declare_parameter<std::vector<double>>("camera_matrix.data");
        declare_parameter<std::vector<double>>("extrinsic.rotation");
        declare_parameter<std::vector<double>>("extrinsic.translation");

        std::vector<double> k_data, r_data, t_data;
        get_parameter("camera_matrix.data", k_data);
        get_parameter("extrinsic.rotation", r_data);
        get_parameter("extrinsic.translation", t_data);

        K_ = read_matrix3(k_data);
        R_ = read_matrix3(r_data);
        t_ = read_vector3(t_data);

        RCLCPP_INFO(this->get_logger(), "Camera and extrinsic parameters loaded");

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&SensorListener::image_callback, this, std::placeholders::_1));

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&SensorListener::scan_callback, this, std::placeholders::_1));

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/projected_image", 10);
    }

private:
    // Eigen行列変換用
    Eigen::Matrix3d read_matrix3(const std::vector<double>& data) {
        Eigen::Matrix3d mat;
        for (int i = 0; i < 9; ++i)
            mat(i / 3, i % 3) = data[i];
        return mat;
    }

    Eigen::Vector3d read_vector3(const std::vector<double>& data) {
        return Eigen::Vector3d(data[0], data[1], data[2]);
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 最新画像を保存（スレッドセーフにするならmutex必要）
        latest_image_ = msg;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        if (!latest_image_) return;

        // OpenCV画像に変換
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(latest_image_, "bgr8");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        double angle = msg->angle_min;
        for (size_t i = 0; i < msg->ranges.size(); ++i, angle += msg->angle_increment) {
            double r = msg->ranges[i];
            if (!std::isfinite(r)) continue;

            Eigen::Vector3d p_lidar(r * cos(angle), r * sin(angle), 0.0);
            Eigen::Vector3d p_cam = R_ * p_lidar + t_;

            if (p_cam.z() <= 0) continue;

            Eigen::Vector3d p_img = K_ * p_cam;
            int u = static_cast<int>(p_img.x() / p_img.z());
            int v = static_cast<int>(p_img.y() / p_img.z());

            if (u >= 0 && u < cv_ptr->image.cols && v >= 0 && v < cv_ptr->image.rows) {
                cv::circle(cv_ptr->image, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), -1);
            }
        }

        auto out_msg = cv_ptr->toImageMsg();
        out_msg->header = msg->header; // or use image header
        image_pub_->publish(*out_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    sensor_msgs::msg::Image::SharedPtr latest_image_;

    Eigen::Matrix3d K_, R_;
    Eigen::Vector3d t_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SensorListener>());
    rclcpp::shutdown();
    return 0;
}
