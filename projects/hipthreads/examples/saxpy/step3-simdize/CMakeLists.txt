cmake_minimum_required(VERSION 3.21)

project(Saxpy LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

add_executable(saxpy main.hip)

target_compile_options(saxpy PUBLIC "-O3" "-fgpu-rdc" "-march=native")
target_link_options(saxpy PUBLIC "-fgpu-rdc")

find_package(hipthreads)
find_package(rocprim REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)

target_link_libraries(saxpy PRIVATE hipthreads::hipthreads)
target_link_libraries(saxpy PRIVATE roc::rocthrust)

# needed to get intellisense to stop complaining about the co-ordinate built-ins not being defined
target_compile_definitions(saxpy PUBLIC __HIP__)

# cd examples/saxpy/step3-simdize && cmake -B build && cmake --build ./build && ./build/bin/saxpy
