# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu PRIVATE
    AnalysisManager.cpp
    IRBase.cpp
    BasicBlock.cpp
    Function.cpp
    PassManager.cpp
    DebugPrintInstrumentation.cpp
    DAGScheduleJsonWriter.cpp
)
