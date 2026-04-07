cmake_minimum_required(VERSION 3.21)

project(SparseMM LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(spmm main.hip)

target_compile_options(spmm PUBLIC "-O3" "-fgpu-rdc" "-march=native")
target_link_options(spmm PUBLIC "-fgpu-rdc")

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

find_package(hipthreads)
find_package(rocprim REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)

target_link_libraries(spmm PRIVATE hipthreads::hipthreads)
target_link_libraries(spmm PRIVATE roc::rocthrust)

# If you haven't already, pull the matrices from git lfs:
# sudo apt install git-lfs
# git lfs pull examples/sparse-mat-mul/data/*

# then build & run:
# cd examples/sparse-mat-mul/step3-hipthread-port && cmake -B build && cmake --build ./build && ./build/bin/spmm
