# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu PRIVATE
    IntrinsicExpansionPass.cpp
    ToStinkyAsmPass.cpp
    CompositeInstructionLoweringPass.cpp
    LogicalPeepholePass.cpp
)
