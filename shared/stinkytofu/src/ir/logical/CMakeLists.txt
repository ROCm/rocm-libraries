# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu PRIVATE
    LogicalToFunctionConverter.cpp
    LogicalOpcode.cpp
    IntrinsicLibrary.cpp
    IntrinsicRegistry.cpp
    IntrinsicPatternConverter.cpp
)
