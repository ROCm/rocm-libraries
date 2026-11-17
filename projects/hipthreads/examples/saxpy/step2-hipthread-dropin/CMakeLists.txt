cmake_minimum_required(VERSION 3.21)

project(Saxpy LANGUAGES CXX HIP)

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

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

add_executable(saxpy main.hip)

target_compile_options(saxpy PUBLIC "-O3" "-fgpu-rdc" "-march=native")
target_link_options(saxpy PUBLIC "-fgpu-rdc" "--hip-link")

if(WIN32)
  # TheRock embeds /DEFAULTLIB:c++ in amdhip64.lib but doesn't ship libc++ on Windows,
  # and clang's HIP driver auto-injects -lc++. Suppress both; MSVC's stdlib pulls in
  # automatically via #pragma comment(lib, "msvcprt") from MSVC headers.
  target_link_options(saxpy PRIVATE "-nostdlib++" "-Xlinker" "/NODEFAULTLIB:c++.lib")
endif()

find_package(hipthreads REQUIRED)
find_package(rocprim REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)

target_link_libraries(saxpy PRIVATE hipthreads::hipthreads)
target_link_libraries(saxpy PRIVATE roc::rocthrust)

# needed to get intellisense to stop complaining about the co-ordinate built-ins not being defined
target_compile_definitions(saxpy PUBLIC __HIP__)

# LINUX:
# cd examples/saxpy/step2-hipthread-dropin && cmake -B build && cmake --build ./build && ./build/bin/saxpy

# WINDOWS (from x64 Native Tools Command Prompt for VS 2022):
# cd examples/saxpy/step2-hipthread-dropin && cmake -B build -G Ninja -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_C_COMPILER="clang" -DHIP_PLATFORM=amd -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=Release && cmake --build build && build\bin\saxpy.exe
