cmake_minimum_required(VERSION 3.8)
project(action_filter)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ackermann_msgs REQUIRED)

add_executable(action_filter_node src/action_filter_node.cpp)
ament_target_dependencies(action_filter_node rclcpp ackermann_msgs)
install(TARGETS action_filter_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME})  

ament_package()

