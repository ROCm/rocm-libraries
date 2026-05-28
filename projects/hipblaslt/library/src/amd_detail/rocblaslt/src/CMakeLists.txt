# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT



target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Debug.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_auxiliary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_mat.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/status.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensile_host.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/UserDrivenTuningParser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_transform.cpp"
)
target_include_directories(hipblaslt PRIVATE "${PROJECT_SOURCE_DIR}/device-library")

# FP64 emulation via Ozaki Scheme II (INT8 tensor cores + CRT).
# This file contains __global__ HIP kernels and must be compiled as HIP,
# even though the rest of the sources in this directory are plain C++.
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/fp64_emulation.cpp"
    PROPERTIES LANGUAGE HIP
)
target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/fp64_emulation.cpp"
)
if(HIPBLASLT_ENABLE_ROCROLLER)
    add_subdirectory(rocroller)
endif()
