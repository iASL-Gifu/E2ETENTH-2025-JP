cmake_minimum_required(VERSION 3.8)
project(camera_lidar_calib)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(camera_lidar_calib_node src/camera_lidar_calib_node.cpp)
ament_target_dependencies(camera_lidar_calib_node 
  rclcpp
  sensor_msgs
  OpenCV
  cv_bridge
  Eigen3
)

install(TARGETS camera_lidar_calib_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME})  

ament_package()

