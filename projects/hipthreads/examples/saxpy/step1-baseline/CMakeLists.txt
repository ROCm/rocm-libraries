cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_COMPILER clang++)

project(Saxpy LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(saxpy main.cxx)

target_compile_options(saxpy PUBLIC "-O3" "-march=native")

find_package(Threads REQUIRED)
target_link_libraries(saxpy PRIVATE Threads::Threads)

# LINUX:
# cd examples/saxpy/step1-baseline && cmake -B build && cmake --build ./build && ./build/bin/saxpy

# WINDOWS (from x64 Native Tools Command Prompt for VS 2022):
# cd examples/saxpy/step1-baseline && cmake -B build -G Ninja -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_C_COMPILER="clang" -DCMAKE_BUILD_TYPE=Release && cmake --build build && build\bin\saxpy.exe
