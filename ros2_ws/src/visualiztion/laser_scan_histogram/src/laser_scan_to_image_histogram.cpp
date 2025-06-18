#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h> // cv_bridgeのヘッダ
#include <opencv2/opencv.hpp>    // OpenCVのヘッダ

class LaserScanToImageHistogram : public rclcpp::Node
{
public:
  LaserScanToImageHistogram()
  : Node("laser_scan_to_image_histogram")
  {
    // パラメータの宣言とデフォルト値の設定
    this->declare_parameter<int>("image_width", 800);
    this->declare_parameter<int>("image_height", 400);
    this->declare_parameter<double>("max_range", 10.0); // ヒストグラムの最大距離（表示範囲）
    this->declare_parameter<double>("min_range", 0.0);  // ヒストグラムの最小距離（表示範囲）
    this->declare_parameter<int>("bar_width_pixels", 1); // 各棒のピクセル幅

    subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", // LiDARスキャンデータのトピック名
      10,
      std::bind(&LaserScanToImageHistogram::laserScanCallback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/histogram_image", 10);

    RCLCPP_INFO(this->get_logger(), "LaserScan to Image Histogram Node has been started.");
  }

private:
  void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    // パラメータの取得
    int img_width = this->get_parameter("image_width").as_int();
    int img_height = this->get_parameter("image_height").as_int();
    double max_range = this->get_parameter("max_range").as_double();
    double min_range = this->get_parameter("min_range").as_double();
    int bar_width_pixels = this->get_parameter("bar_width_pixels").as_int();

    // ヒストグラム用の画像を作成 (背景は黒)
    cv::Mat hist_image(img_height, img_width, CV_8UC3, cv::Scalar(0, 0, 0));

    // LiDARスキャンデータのインデックスと距離を使ってヒストグラムを描画
    // 画像の横幅をスキャン点数に合わせて調整するか、あるいはスキャン点数が多い場合はバーを間引く
    // ここでは単純にスキャン点のインデックスをX軸にマッピング
    // X軸のスケールは img_width / msg->ranges.size()
    // Y軸のスケールは img_height / max_range

    double x_scale = static_cast<double>(img_width) / msg->ranges.size();
    double y_scale = static_cast<double>(img_height) / (max_range - min_range);

    for (size_t i = 0; i < msg->ranges.size(); ++i) {
      float distance = msg->ranges[i];

      // 無効な距離値（infやnan）はスキップ
      if (std::isinf(distance) || std::isnan(distance)) {
        continue;
      }

      // 距離値を指定範囲にクランプ
      double clamped_distance = std::max(min_range, std::min(max_range, static_cast<double>(distance)));

      // ヒストグラムの棒の位置と高さ
      int x_start = static_cast<int>(i * x_scale);
      int x_end = static_cast<int>((i + 1) * x_scale); // bar_width_pixelsで棒の幅を制御する場合
      if (bar_width_pixels > 0) { // パラメータで指定された棒の幅を使う場合
          x_end = x_start + bar_width_pixels;
      }
      x_end = std::min(x_end, img_width); // 画像の幅を超えないようにクランプ

      int bar_height_pixels = static_cast<int>((clamped_distance - min_range) * y_scale);
      bar_height_pixels = std::min(bar_height_pixels, img_height); // 画像の高さを超えないようにクランプ

      // 棒の描画 (矩形を描画)
      // 色は緑色 (BGR形式なので 0, 255, 0)
      // 棒の描画は画像の底辺から上に向かって描画されるようにY座標を計算
      cv::Rect bar_rect(x_start, img_height - bar_height_pixels, x_end - x_start, bar_height_pixels);
      cv::rectangle(hist_image, bar_rect, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    // 画像の枠線を描画 (オプション)
    cv::rectangle(hist_image, cv::Rect(0, 0, img_width, img_height), cv::Scalar(255, 255, 255), 1);

    // X軸とY軸のラベルを描画 (簡略化のためここでは省略。必要であればOpenCVのputTextを使用)

    // OpenCVのMatをROSのImageメッセージに変換
    std_msgs::msg::Header header;
    header.stamp = this->now();
    header.frame_id = "laser_frame"; // またはLiDARのフレームID

    sensor_msgs::msg::Image::SharedPtr image_msg = cv_bridge::CvImage(header, "bgr8", hist_image).toImageMsg();
    publisher_->publish(*image_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaserScanToImageHistogram>());
  rclcpp::shutdown();
  return 0;
}