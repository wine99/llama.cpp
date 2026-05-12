cmake_minimum_required(VERSION 3.16)
project(ov_kv_cache_reproducer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED)

add_executable(reproducer reproducer.cpp)
target_link_libraries(reproducer PRIVATE openvino::runtime)

add_executable(scatter_reproducer scatter_reproducer.cpp)
target_link_libraries(scatter_reproducer PRIVATE openvino::runtime)
