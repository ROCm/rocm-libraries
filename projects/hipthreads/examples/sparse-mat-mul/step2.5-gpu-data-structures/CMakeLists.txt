cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_COMPILER clang++)

project(SparseMM LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(spmm main.cxx)
set_source_files_properties(main.cxx PROPERTIES LANGUAGE HIP)

target_compile_options(spmm PUBLIC "-O3" "-march=native")
target_compile_definitions(spmm PUBLIC _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

find_package(Threads REQUIRED)
find_package(rocthrust REQUIRED)
find_package(libhipcxx REQUIRED)

target_link_libraries(spmm PRIVATE Threads::Threads)
target_link_libraries(spmm PRIVATE roc::rocthrust)
target_link_libraries(spmm PRIVATE libhipcxx::libhipcxx)

# If you haven't already, pull the matrices from git lfs:
# sudo apt install git-lfs
# git lfs pull examples/sparse-mat-mul/data/*

# then build & run:
# LINUX:
# cd examples/sparse-mat-mul/step2.5-gpu-data-structures && cmake -B build && cmake --build build && ./build/bin/spmm

# WINDOWS (from x64 Native Tools Command Prompt for VS 2022):
# cd examples/sparse-mat-mul/step2.5-gpu-data-structures && cmake -B build -G Ninja -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_C_COMPILER="clang" -DHIP_PLATFORM=amd -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=Release . && cmake --build build && build\bin\spmm.exe
