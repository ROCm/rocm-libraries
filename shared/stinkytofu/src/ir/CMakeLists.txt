# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    DumpStinkyFunctionPass.cpp
)

add_subdirectory(logical)
add_subdirectory(asm)
