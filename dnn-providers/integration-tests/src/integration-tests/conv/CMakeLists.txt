# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(${INTEGRATION_TESTS_EXE}
    PRIVATE
        IntegrationConvForwardSerializeRoundTrip.cpp
)

if(BUILD_CPP_GRAPH_TESTS)
    target_sources(${INTEGRATION_TESTS_EXE}
        PRIVATE
            IntegrationGpuConvForward.cpp
            IntegrationGpuConvBackwardData.cpp
            IntegrationGpuConvBackwardWeights.cpp
            IntegrationGpuConvFwdBiasActiv.cpp
    )
endif()
