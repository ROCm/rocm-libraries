# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(${INTEGRATION_TESTS_EXE}
    PRIVATE
        IntegrationGoldenRefSdpaFwdInference.cpp
)

if(BUILD_CPP_GRAPH_TESTS)
    target_sources(${INTEGRATION_TESTS_EXE}
        PRIVATE
            IntegrationGpuSdpaBackward.cpp
            IntegrationGpuSdpaFwdInference.cpp
    )
endif()
