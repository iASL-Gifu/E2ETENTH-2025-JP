#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class ScanOverlayNode : public rclcpp::Node {
public:
    ScanOverlayNode() : Node("scan_overlay_node") {
        this->declare_parameter<int>("scan_buffer_size", 30);    // スキャンバッファの最大サイズ
        this->declare_parameter<double>("max_time_difference", 0.05); // 画像とスキャンの最大許容時間差 (秒)

        // パラメータの取得
        scan_buffer_size_ = this->get_parameter("scan_buffer_size").as_int();
        max_time_difference_ = this->get_parameter("max_time_difference").as_double();

        RCLCPP_INFO(this->get_logger(), "スキャンバッファの最大サイズ: %d", scan_buffer_size_);
        RCLCPP_INFO(this->get_logger(), "許容される最大時間差: %.3f 秒", max_time_difference_);
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10, std::bind(&ScanOverlayNode::image_callback, this, std::placeholders::_1));
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ScanOverlayNode::scan_callback, this, std::placeholders::_1));
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    std::deque<sensor_msgs::msg::LaserScan::ConstSharedPtr> scan_buffer_;
    std::mutex buffer_mutex_; // スキャンバッファを保護するためのミューテックス

    int scan_buffer_size_;
    double max_time_difference_;

    cv::Mat latest_image_;
    bool image_received_ = false;

    // カメラ内部パラメータ
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        540.0773, 0.0, 396.42239, 
        0.0, 539.36402, 297.13506, 
        0.0, 0.0, 1.0);

    cv::Mat D = (cv::Mat_<double>(5, 1) << -0.053993, 0.056719, -0.001689, -0.003580, 0.0);

    cv::Mat R = (cv::Mat_<double>(3, 3) << 
    0,  0, 1,
    -1,  0, 0,
     0, -1, 0
    );
    cv::Mat t = (cv::Mat_<double>(3, 1) << 0.00, 0.1, 0.05);
    // cv::Mat t = (cv::Mat_<double>(3, 1) << -0.01122, -0.02711, 0.23163);
    cv::Mat R_lc = R.clone();          // Camera→LiDAR (Tlc) をそのまま
    cv::Mat t_lc = t.clone();          // ↑
    cv::Mat R_cl = R_lc.t();           // ← 転置で LiDAR→Camera
    cv::Mat t_cl = -R_cl * t_lc; 

// LaserScanメッセージをバッファに保存するコールバック
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        std::lock_guard<std::mutex> lock(buffer_mutex_); // ミューテックスでバッファを保護
        
        scan_buffer_.push_back(scan_msg);
        
        // バッファサイズが上限を超えたら、古いデータから削除
        while (scan_buffer_.size() > static_cast<size_t>(scan_buffer_size_)) {
            scan_buffer_.pop_front();
        }
        RCLCPP_DEBUG(this->get_logger(), "スキャン受信: %ld.%09ld, 現在のバッファサイズ: %zu",
            scan_msg->header.stamp.sec, scan_msg->header.stamp.nanosec, scan_buffer_.size());
    }

    // builtin_interfaces::msg::Time を double型 の秒に変換するヘルパー関数
    double timestamp_to_seconds(const builtin_interfaces::msg::Time& stamp) {
        return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
    }

    // Imageメッセージを受信し、最適なLaserScanと同期処理するコールバック
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr image_msg) {
        RCLCPP_DEBUG(this->get_logger(), "画像受信: %ld.%09ld",
            image_msg->header.stamp.sec, image_msg->header.stamp.nanosec);

        sensor_msgs::msg::LaserScan::ConstSharedPtr best_scan_match;
        double min_abs_time_diff = std::numeric_limits<double>::max();

        double image_timestamp_sec = timestamp_to_seconds(image_msg->header.stamp);

        { // ミューテックスのスコープを限定
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            
            if (scan_buffer_.empty()) {
                RCLCPP_WARN(this->get_logger(), "スキャンバッファが空です。画像 %f を処理できません。", image_timestamp_sec);
                // オプション：スキャンなしで画像だけ表示する場合
                // cv::Mat current_image_cv = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
                // cv::imshow("Laser Scan Overlay", current_image_cv);
                // cv::waitKey(1);
                return;
            }

            // バッファ内の全てのスキャンを調べて、画像と最も時間差が小さいものを探す
            for (const auto& scan_in_buffer : scan_buffer_) {
                double scan_timestamp_sec = timestamp_to_seconds(scan_in_buffer->header.stamp);
                double current_abs_diff = std::abs(image_timestamp_sec - scan_timestamp_sec);

                if (current_abs_diff < min_abs_time_diff) {
                    min_abs_time_diff = current_abs_diff;
                    best_scan_match = scan_in_buffer;
                }
            }
        } // ミューテックスのスコープ終了

        // 最適なスキャンが見つかったか、かつ時間差が許容範囲内かを確認
        if (!best_scan_match) {
             RCLCPP_WARN(this->get_logger(), "何らかの理由で最適なスキャンが見つかりませんでした。");
            return; // 通常ここには来ないはず (バッファが空でなければ)
        }
        
        if (min_abs_time_diff > max_time_difference_) {
            RCLCPP_WARN(this->get_logger(), "画像 (%f) に対する最適スキャン (%f) との時間差 %.3fs が許容範囲 %.3fs を超えています。",
                image_timestamp_sec, timestamp_to_seconds(best_scan_match->header.stamp),
                min_abs_time_diff, max_time_difference_);
            // オプション：スキャンなしで画像だけ表示する場合
            // cv::Mat current_image_cv = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
            // cv::imshow("Laser Scan Overlay", current_image_cv);
            // cv::waitKey(1);
            return;
        }

        RCLCPP_INFO(this->get_logger(), "マッチ成功: 画像 %f と スキャン %f (時間差 %.6f 秒)",
            image_timestamp_sec, timestamp_to_seconds(best_scan_match->header.stamp), min_abs_time_diff);

        // --- ここから先の点群投影と描画処理は、以前の synchronized_callback と同様 ---
        cv::Mat current_image_cv;
        try {
            current_image_cv = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat undistorted_image;
        cv::undistort(current_image_cv, undistorted_image, K, D);

        double angle = best_scan_match->angle_min;
        for (size_t i = 0; i < best_scan_match->ranges.size(); ++i, angle += best_scan_match->angle_increment) {
            float r = best_scan_match->ranges[i];
            if (r < best_scan_match->range_min || r > best_scan_match->range_max || std::isnan(r) || std::isinf(r)) {
                continue;
            }
            double x_lidar = r * std::cos(angle);
            double y_lidar = r * std::sin(angle);
            double z_lidar = 0.0; 
            cv::Mat p_lidar = (cv::Mat_<double>(3, 1) << x_lidar, y_lidar, z_lidar);
            
            cv::Mat p_cam = R_cl * p_lidar + t_cl;
            double p_cam_z = p_cam.at<double>(2, 0);

            if (p_cam_z <= 1e-3) { // 非常に近い点やマイナスの奥行きを除外
                continue; 
            }
            
            cv::Point2d uv;
            uv.x = (p_cam.at<double>(0, 0) / p_cam_z) * K.at<double>(0, 0) + K.at<double>(0, 2);
            uv.y = (p_cam.at<double>(1, 0) / p_cam_z) * K.at<double>(1, 1) + K.at<double>(1, 2);
            
            if (uv.x >= 0 && uv.x < undistorted_image.cols && uv.y >= 0 && uv.y < undistorted_image.rows) {
                cv::circle(undistorted_image, uv, 2, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::imshow("Laser Scan Overlay", undistorted_image);
        cv::waitKey(1);
    }


};
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ScanOverlayNode>());
    rclcpp::shutdown();
    return 0;
}
