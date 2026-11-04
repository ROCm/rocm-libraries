# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    AsmVerifierPass.cpp
    GirFrameAnalysis.cpp
    HazardGapAnalysisPass.cpp
)

add_subdirectory(ssa)
