# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    LiftAsmRegistersToSSAPass.cpp
    DumpCanonicalSSAPass.cpp
    CanonicalSSADestruction.cpp
    ReplayLegacyColoringPass.cpp
)
