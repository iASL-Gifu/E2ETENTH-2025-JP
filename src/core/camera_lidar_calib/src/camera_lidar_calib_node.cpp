#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.hpp>

class ScanOverlayNode : public rclcpp::Node, public std::enable_shared_from_this<ScanOverlayNode> {
public:
    ScanOverlayNode() : Node("scan_overlay_node") {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10, std::bind(&ScanOverlayNode::image_callback, this, std::placeholders::_1));
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ScanOverlayNode::scan_callback, this, std::placeholders::_1));

        this->declare_parameter<std::vector<double>>("camera_matrix.data", std::vector<double>{});
        this->declare_parameter<std::vector<double>>("distortion_coefficients.data", std::vector<double>{});
        this->declare_parameter<std::vector<double>>("extrinsic.rotation", std::vector<double>{});
        this->declare_parameter<std::vector<double>>("extrinsic.translation", std::vector<double>{});

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

        it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
        image_pub_ = it_->advertise("overlay_image", 1);

        RCLCPP_INFO(this->get_logger(), "Parameters successfully loaded and publisher initialized.");
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

        // パブリッシュ用のROSメッセージに変換して送信
        std_msgs::msg::Header header = msg->header;
        sensor_msgs::msg::Image::SharedPtr output_msg =
            cv_bridge::CvImage(header, "bgr8", undistorted).toImageMsg();
        image_pub_.publish(output_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    sensor_msgs::msg::LaserScan::ConstSharedPtr last_scan_;

    cv::Mat K_, D_, R_cl_, t_cl_;

    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Publisher image_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ScanOverlayNode>());
    rclcpp::shutdown();
    return 0;
}
