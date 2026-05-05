# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_POOL2D_BWD "tile_example_pool2d_bwd")
message(DEBUG "adding example ${EXAMPLE_POOL2D_BWD}")

add_executable(${EXAMPLE_POOL2D_BWD} pool2d_bwd.cpp)
target_include_directories(${EXAMPLE_POOL2D_BWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_compile_options(${EXAMPLE_POOL2D_BWD} PRIVATE ${EXAMPLE_POOL_COMPILE_OPTIONS})
