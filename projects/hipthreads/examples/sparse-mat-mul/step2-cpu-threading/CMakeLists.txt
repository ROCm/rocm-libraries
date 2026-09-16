cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_COMPILER clang++)

project(SparseMM LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(spmm main.cxx)

target_compile_options(spmm PUBLIC "-O3" "-march=native")

find_package(Threads REQUIRED)
target_link_libraries(spmm PRIVATE Threads::Threads)

# If you haven't already, pull the matrices from git lfs:
# sudo apt install git-lfs
# git lfs pull examples/sparse-mat-mul/data/*

# then build & run:
# cd examples/sparse-mat-mul/step2-cpu-threading && cmake -B build && cmake --build build && ./build/bin/spmm

# WINDOWS (from x64 Native Tools Command Prompt for VS 2022):
# cd examples/sparse-mat-mul/step2-cpu-threading && cmake -B build -G Ninja -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_C_COMPILER="clang" -DCMAKE_BUILD_TYPE=Release . && cmake --build build && build\bin\spmm.exe
