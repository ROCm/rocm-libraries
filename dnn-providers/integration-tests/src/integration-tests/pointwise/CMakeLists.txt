# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(BUILD_CPP_GRAPH_TESTS)
    target_sources(${INTEGRATION_TESTS_EXE}
        PRIVATE
            IntegrationGpuActivationForward.cpp
    )
endif()
