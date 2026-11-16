# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    PassBuilder.cpp
    RegisterAllocationPipeline.cpp
)

add_subdirectory(backend)
