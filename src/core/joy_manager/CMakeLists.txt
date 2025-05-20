cmake_minimum_required(VERSION 3.8)
project(joy_manager)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)

add_executable(joy_manager_node src/joy_manager_node.cpp)
ament_target_dependencies(joy_manager_node rclcpp std_msgs sensor_msgs ackermann_msgs)

install(TARGETS joy_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME})  

ament_package()

