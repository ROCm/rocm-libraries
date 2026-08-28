# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shares the switch with 30_hipblaslt_a2a_sdma.
option(HIPBLASLT_ENABLE_A2A_SDMA_SAMPLE
    "Build the fused GEMM+A2A SDMA sample (needs hsakmt)" OFF)

if(NOT HIPBLASLT_ENABLE_A2A_SDMA_SAMPLE OR WIN32)
    return()
endif()

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

if(NOT TARGET numa::numa)
    find_library(NUMA_LIBRARY
        NAMES numa
        HINTS "${ROCM_PATH}/lib/rocm_sysdeps/lib"
        PATHS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
    find_path(NUMA_INCLUDE_DIR
        NAMES numa.h
        HINTS "${ROCM_PATH}/include/rocm_sysdeps/include"
        PATHS "${ROCM_PATH}/include/rocm_sysdeps/include")
    if(NUMA_LIBRARY)
        add_library(numa::numa UNKNOWN IMPORTED)
        set_target_properties(numa::numa PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}")
        if(NUMA_INCLUDE_DIR)
            set_target_properties(numa::numa PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
        endif()
    else()
        message(FATAL_ERROR
            "libnuma not found but required by hsakmt::hsakmt for "
            "HIPBLASLT_ENABLE_A2A_SDMA_SAMPLE. Searched the ROCm-vendored path "
            "'${ROCM_PATH}/lib/rocm_sysdeps/lib' and the default system library "
            "paths. Set ROCM_PATH if your ROCm is elsewhere, or point "
            "NUMA_LIBRARY at the libnuma.so to use.")
    endif()
endif()

find_package(hsa-runtime64 REQUIRED)
find_package(hsakmt REQUIRED)

if(TARGET hsakmt::hsakmt)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../../tensilelite/cmake/HsakmtLinkInterface.cmake")
    get_target_property(_hsakmt_ill hsakmt::hsakmt INTERFACE_LINK_LIBRARIES)
    if(_hsakmt_ill)
        tensilelite_sanitize_hsakmt_link_interface(_hsakmt_ill_clean
            "${ROCM_PATH}/lib/rocm_sysdeps/lib" _hsakmt_ill)
        set_target_properties(hsakmt::hsakmt PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_hsakmt_ill_clean}")
    endif()
endif()

add_executable(sample_a2a_sdma_multiprocess
    "${CMAKE_CURRENT_SOURCE_DIR}/sample_a2a_sdma_multiprocess.cpp")
target_link_libraries(sample_a2a_sdma_multiprocess
    PRIVATE sample-helper hsakmt::hsakmt hsa-runtime64::hsa-runtime64)

# Per-source: <SdmaQueue.hpp> lives in the TensileLite client include dir.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_a2a_sdma_multiprocess.cpp
    PROPERTIES
        COMPILE_OPTIONS
            "-I${CMAKE_CURRENT_SOURCE_DIR}/../../../tensilelite/client/include"
)
