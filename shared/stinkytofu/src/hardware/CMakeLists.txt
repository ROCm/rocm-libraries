# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# GfxXXX.hpp are generated in build/hardware/arch_headers/ by hardware/CMakeLists.txt
# ArchHelper_includes.inc includes them directly (same dir)
target_sources(stinkytofu PRIVATE
    ArchHelper.cpp
    ComgrProbe.cpp
    HwReg.cpp
    ToolchainCaps.cpp
)
