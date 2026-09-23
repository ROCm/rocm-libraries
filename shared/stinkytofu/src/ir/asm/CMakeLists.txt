# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    AsmSetSymbolMap.cpp
    SymbolicRegName.cpp
    StinkyAsmIR.cpp
    StinkyModifiers.cpp
    StinkySignature.cpp
)

add_subdirectory(ssa)
