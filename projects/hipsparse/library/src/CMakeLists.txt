# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Private include directory
target_include_directories(hipsparse PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Backend-specific subdirectories
if(HIPSPARSE_ENABLE_CUDA)
    add_subdirectory(nvidia_detail)
elseif(HIPSPARSE_ENABLE_HIP)
    add_subdirectory(amd_detail)
endif()
