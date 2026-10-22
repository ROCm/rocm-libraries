# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT MOSAIC_BUILD_HIP_EXAMPLES)
    return()
endif()

add_executable(check_tiling check_tiling.cpp)
set_source_files_properties(check_tiling.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(check_tiling PRIVATE mosaic hip::host)
