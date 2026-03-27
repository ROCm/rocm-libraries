# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu PRIVATE
    IRBase.cpp
    BasicBlock.cpp
    Function.cpp
    PassManager.cpp
    DAGScheduleJsonWriter.cpp
)
