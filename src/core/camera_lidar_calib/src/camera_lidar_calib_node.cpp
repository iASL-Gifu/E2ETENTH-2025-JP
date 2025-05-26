#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ScanOverlayNode : public rclcpp::Node {
public:
    ScanOverlayNode() : Node("scan_overlay_node") {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10, std::bind(&ScanOverlayNode::image_callback, this, std::placeholders::_1));
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ScanOverlayNode::scan_callback, this, std::placeholders::_1));

        // パラメータ取得（launchから読み込まれるparamファイルに依存）
        std::vector<double> k_data = this->get_parameter("camera_matrix.data").as_double_array();
        std::vector<double> d_data = this->get_parameter("distortion_coefficients.data").as_double_array();
        std::vector<double> r_data = this->get_parameter("extrinsic.rotation").as_double_array();
        std::vector<double> t_data = this->get_parameter("extrinsic.translation").as_double_array();

        K_ = cv::Mat(3, 3, CV_64F, k_data.data()).clone();
        D_ = cv::Mat(d_data).reshape(1, 1).clone();

        cv::Mat R_lc(3, 3, CV_64F, r_data.data());
        cv::Mat t_lc(3, 1, CV_64F, t_data.data());
        R_cl_ = R_lc.t();
        t_cl_ = -R_cl_ * t_lc;

        RCLCPP_INFO(this->get_logger(), "Parameters successfully loaded from ROS param server.");
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg) {
        last_scan_ = msg;
    }

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
        if (!last_scan_) return;

        cv::Mat image;
        try {
            image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        cv::Mat undistorted;
        cv::undistort(image, undistorted, K_, D_);

        double angle = last_scan_->angle_min;
        for (size_t i = 0; i < last_scan_->ranges.size(); ++i, angle += last_scan_->angle_increment) {
            float r = last_scan_->ranges[i];
            if (!std::isfinite(r)) continue;

            // LiDAR座標点
            cv::Mat p_lidar = (cv::Mat_<double>(3, 1) << r * cos(angle), r * sin(angle), 0.0);
            cv::Mat p_cam = R_cl_ * p_lidar + t_cl_;

            double z = p_cam.at<double>(2, 0);
            if (z <= 1e-3) continue;

            double x = p_cam.at<double>(0, 0) / z * K_.at<double>(0, 0) + K_.at<double>(0, 2);
            double y = p_cam.at<double>(1, 0) / z * K_.at<double>(1, 1) + K_.at<double>(1, 2);

            if (x >= 0 && x < undistorted.cols && y >= 0 && y < undistorted.rows) {
                cv::circle(undistorted, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::imshow("Scan Overlay", undistorted);
        cv::waitKey(1);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    sensor_msgs::msg::LaserScan::ConstSharedPtr last_scan_;

    cv::Mat K_, D_, R_cl_, t_cl_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ScanOverlayNode>());
    rclcpp::shutdown();
    return 0;
}