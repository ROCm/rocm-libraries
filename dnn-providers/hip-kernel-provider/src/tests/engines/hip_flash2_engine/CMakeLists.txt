# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHipFlash2Engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHipFlash2FwdPlanBuilder.cpp
)
