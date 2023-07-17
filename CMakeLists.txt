cmake_minimum_required(VERSION 3.0.2)
project(tensorrt_yolov8)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(catkin REQUIRED COMPONENTS
  roslaunch
  sensor_msgs
  std_msgs
  roscpp
  rospy
  roslib
  cv_bridge
  image_transport
  autoware_perception_msgs
)

# We require CUDA, OpenCV, and TensorRT
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

## find tensorrt and set variables TensorRT_INCLUDE_DIRS and TensorRT_LIBRARIES
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(TensorRT REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs cv_bridge image_transport autoware_perception_msgs
  DEPENDS OpenCV TensorRT CUDA
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/main.cpp src/engineRosWrapper.cpp src/engineTool.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})