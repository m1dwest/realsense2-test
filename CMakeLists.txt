cmake_minimum_required(VERSION 3.16)

project(realsense2_test CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(realsense2 CONFIG REQUIRED)
find_package(lz4 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

add_executable(bin main.cpp)

target_link_libraries(bin PRIVATE realsense2::realsense2 lz4::lz4 ${OpenCV_LIBS})
