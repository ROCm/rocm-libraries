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

# LINUX:
# cd examples/InOneWeekendRaytracer/step1-baseline && cmake -B build && cmake --build build && time ./build/bin/inOneWeekend > image.ppm

# WINDOWS (from x64 Native Tools Command Prompt for VS 2022):
# cd examples/InOneWeekendRaytracer/step1-baseline && cmake -G Ninja -B build -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_C_COMPILER="clang" -DCMAKE_BUILD_TYPE=Release . && cmake --build build && build\bin\inOneWeekend > image.ppm
