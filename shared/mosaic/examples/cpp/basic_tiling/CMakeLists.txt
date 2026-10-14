# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT MOSAIC_BUILD_HIP_EXAMPLES)
    return()
endif()

add_executable(check_tiling check_tiling.cpp)
target_link_libraries(check_tiling PRIVATE mosaic)
