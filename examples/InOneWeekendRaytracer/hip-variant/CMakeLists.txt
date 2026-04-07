cmake_minimum_required(VERSION 3.21)

project(RTWeekend LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(inOneWeekend main.hip)

target_compile_options(inOneWeekend PUBLIC "-O3" "-march=native")

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

# cd examples/InOneWeekendRaytracer/hip-variant && cmake -B build && cmake --build build && time ./build/bin/inOneWeekend > image.ppm
