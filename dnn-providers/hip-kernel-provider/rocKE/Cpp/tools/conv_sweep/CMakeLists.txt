# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocke_conv_sweep -- standalone conv candidate sweep tool.
# Enumerates, JIT-compiles, and times implicit-GEMM conv candidates
# using the rocke C API + comgr + HIP.
#
# Not built by default; built at sweep-job time inside the container:
#   cmake -S <rocKE> -B build -DROCKE_BUILD_CONV_SWEEP=ON && cmake --build build -t rocke_conv_sweep

find_package(hip QUIET)
find_library(AMD_COMGR_LIB amd_comgr HINTS /opt/rocm/lib)
find_path(AMD_COMGR_INCLUDE amd_comgr/amd_comgr.h HINTS /opt/rocm/include)

if(NOT hip_FOUND OR NOT AMD_COMGR_LIB OR NOT AMD_COMGR_INCLUDE)
  message(STATUS "rocke_conv_sweep: skipping (requires HIP + amd_comgr)")
  return()
endif()

add_executable(rocke_conv_sweep
    conv_candidate_sweep.cpp
    main.cpp)

target_link_libraries(rocke_conv_sweep PRIVATE
    rocke_core
    ${AMD_COMGR_LIB}
    hip::host
    Threads::Threads)

target_include_directories(rocke_conv_sweep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AMD_COMGR_INCLUDE})

find_package(Threads REQUIRED)

# Standalone helper binary for isolated kernel timing (fork+exec target).
# Depends only on HIP — no rocke_core, no comgr.
add_executable(rocke_kern_time rocke_kern_time.cpp)
target_link_libraries(rocke_kern_time PRIVATE hip::host)
target_include_directories(rocke_kern_time PRIVATE ${AMD_COMGR_INCLUDE})
