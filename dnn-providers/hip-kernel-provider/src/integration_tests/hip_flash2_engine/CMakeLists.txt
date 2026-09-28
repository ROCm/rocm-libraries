# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hip_kernel_provider_integration_tests PRIVATE
    # IntegrationGoldenRefHipFlash2FwdInference.cpp excluded: requires hipdnn-integration-tests project headers
    # (harness/GoldenReferenceCpu.hpp from dnn-providers/integration-tests/src/harness/)
    # Relocate alongside integration-tests/src/integration-tests/sdpa/ to enable.
    ${CMAKE_CURRENT_SOURCE_DIR}/IntegrationGpuHipFlash2Forward.cpp
)
