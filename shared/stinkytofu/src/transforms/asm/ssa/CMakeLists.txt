# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    AsmRegisterLifting.cpp
    LiftAsmRegistersToSSAPass.cpp
    SSADestruction.cpp
)
