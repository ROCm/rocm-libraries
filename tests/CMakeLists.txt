cmake_minimum_required(VERSION 3.11.0)
project(installation-tests LANGUAGES CXX)

find_package(hipblas-common REQUIRED)

add_executable(hello-roc main.cpp)
target_link_libraries(hello-roc PRIVATE roc::hipblas-common)

add_executable(hello-hip main.cpp)
target_link_libraries(hello-hip PRIVATE hip::hipblas-common)
