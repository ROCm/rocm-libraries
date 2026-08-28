cmake_minimum_required(VERSION 3.21)

project(RTWeekend LANGUAGES CXX HIP)

if(WIN32)
  set(CMAKE_CXX_USING_LINKER_DEFAULT "-fuse-ld=lld")
  set(CMAKE_CXX_USING_LINKER_LLD "-fuse-ld=lld")
  set(CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld")
  set(CMAKE_HIP_USING_LINKER_LLD "-fuse-ld=lld")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(inOneWeekend main.hip)

target_compile_options(inOneWeekend PUBLIC "-O3" "-fgpu-rdc" "-march=native")
target_link_options(inOneWeekend PUBLIC "-fgpu-rdc" "--hip-link")

if(WIN32)
  # TheRock embeds /DEFAULTLIB:c++ in amdhip64.lib but doesn't ship libc++ on Windows,
  # and clang's HIP driver auto-injects -lc++. Suppress both; MSVC's stdlib pulls in
  # automatically via #pragma comment(lib, "msvcprt") from MSVC headers.
  target_link_options(inOneWeekend PRIVATE "-nostdlib++" "-Xlinker" "/NODEFAULTLIB:c++.lib")
endif()

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

find_package(hipthreads REQUIRED)
find_package(rocprim REQUIRED)
find_package(rocthrust REQUIRED)

target_link_libraries(inOneWeekend PRIVATE hipthreads::hipthreads)
target_link_libraries(inOneWeekend PRIVATE roc::rocthrust)

# needed to get intellisense to stop complaining about the co-ordinate built-ins not being defined
target_compile_definitions(inOneWeekend PUBLIC __HIP__)

# LINUX:
# cd examples/InOneWeekendRaytracer/step3-hipthread-dropin && cmake -B build && cmake --build build && time ./build/bin/inOneWeekend > image.ppm

# WINDOWS (from x64 Native Tools Command Prompt for VS 2022):
# cd examples/InOneWeekendRaytracer/step3-hipthread-dropin && cmake -G Ninja -B build -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_C_COMPILER="clang" -DHIP_PLATFORM=amd -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=Release . && cmake --build build && build\bin\inOneWeekend > image.ppm
