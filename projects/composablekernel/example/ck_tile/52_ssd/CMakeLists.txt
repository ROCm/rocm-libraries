# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(tile_example_ssd_fwd example_ssd_fwd.cpp)
target_include_directories(tile_example_ssd_fwd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
