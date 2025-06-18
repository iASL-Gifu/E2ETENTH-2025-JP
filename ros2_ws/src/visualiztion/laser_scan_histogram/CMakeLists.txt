cmake_minimum_required(VERSION 3.8)
project(laser_scan_histogram)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(laser_scan_to_image_histogram src/laser_scan_to_image_histogram.cpp)
ament_target_dependencies(laser_scan_to_image_histogram
  rclcpp
  sensor_msgs
  OpenCV
  cv_bridge
  std_msgs
)
install(TARGETS laser_scan_to_image_histogram
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})  

ament_package()

