# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(${INTEGRATION_TESTS_EXE}
    PRIVATE
        IntegrationCpuSdpaFwdBwdChained.cpp
        IntegrationGoldenRefSdpaFwdInference.cpp
        IntegrationGpuSdpaFwdInference.cpp
)
