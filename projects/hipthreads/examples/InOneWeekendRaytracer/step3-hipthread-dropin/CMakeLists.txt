cmake_minimum_required(VERSION 3.21)

project(RTWeekend LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(inOneWeekend main.hip)

target_compile_options(inOneWeekend PUBLIC "-O3" "-fgpu-rdc" "-march=native")
target_link_options(inOneWeekend PUBLIC "-fgpu-rdc" "--hip-link")

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

find_package(hipthreads REQUIRED)
find_package(rocprim REQUIRED)
find_package(rocthrust REQUIRED)

target_link_libraries(inOneWeekend PRIVATE hipthreads::hipthreads)
target_link_libraries(inOneWeekend PRIVATE roc::rocthrust)

# cd examples/InOneWeekendRaytracer/step3-hipthread-dropin && cmake -B build && cmake --build build && time ./build/bin/inOneWeekend > image.ppm
