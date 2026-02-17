cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_COMPILER clang++)

project(RTWeekend LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(inOneWeekend main.cc)

target_compile_options(inOneWeekend PUBLIC "-O3" "-march=native")

# cd examples/InOneWeekendRaytracer/step1-baseline && cmake -B build && cmake --build build && time ./build/bin/inOneWeekend > image.ppm
