cmake_minimum_required(VERSION 3.8)
project(ackermann_visualizer)

if(CMAKE_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(cv_bridge REQUIRED) 
find_package(image_transport REQUIRED) 
find_package(OpenCV REQUIRED) 

add_executable(ackermann_visualizer_node src/ackermann_visualizer_node.cpp)
ament_target_dependencies(ackermann_visualizer_node
  rclcpp
  std_msgs
  ackermann_msgs
  cv_bridge
  image_transport
  OpenCV
)

# OpenCVのライブラリとインクルードディレクトリを追加
target_link_libraries(ackermann_visualizer_node ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})


install(TARGETS ackermann_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME})  

ament_package()