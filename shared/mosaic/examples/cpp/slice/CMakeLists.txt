# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(slice slice.cpp)
target_link_libraries(slice PRIVATE mosaic)
