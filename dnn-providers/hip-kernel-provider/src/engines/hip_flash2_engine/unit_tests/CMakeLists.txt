# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Unit tests for HipFlash2Engine (>80% code coverage required per hipDNN testing policy).
# Follows the asm_sdpa_engine test pattern.

target_sources(hip_kernel_provider_tests
    PRIVATE
        TestHipFlash2Engine.cpp
        TestHipFlash2FwdPlanBuilder.cpp
)
