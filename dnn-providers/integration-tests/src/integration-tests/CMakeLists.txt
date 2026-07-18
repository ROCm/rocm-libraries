# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

option(BUILD_CPP_GRAPH_TESTS
    "Build the legacy C++ graph integration tests (OFF when bundles are the CI driver)"
    OFF)

target_sources(${INTEGRATION_TESTS_EXE}
    PRIVATE
        IntegrationIsSupportedExtPerformance.cpp
        batchnorm/IntegrationGoldenRefBatchnormFwdInference.cpp
        conv/IntegrationConvForwardSerializeRoundTrip.cpp
        sdpa/IntegrationGoldenRefSdpaFwdInference.cpp
)

# Matmul has not been migrated to bundles yet.
add_subdirectory(matmul)

if(BUILD_CPP_GRAPH_TESTS)
    add_subdirectory(batchnorm)
    add_subdirectory(conv)
    add_subdirectory(layernorm)
    add_subdirectory(reduction)
    add_subdirectory(rmsnorm)
    add_subdirectory(sdpa)
    add_subdirectory(pointwise)
endif()
