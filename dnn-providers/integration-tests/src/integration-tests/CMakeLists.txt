# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

option(BUILD_CPP_GRAPH_TESTS
    "Build the legacy C++ graph integration tests (OFF when bundles are the CI driver)"
    ON)

target_sources(${INTEGRATION_TESTS_EXE}
    PRIVATE
        IntegrationIsSupportedExtPerformance.cpp
)

add_subdirectory(batchnorm)
add_subdirectory(conv)
add_subdirectory(layernorm)
add_subdirectory(matmul)
add_subdirectory(sdpa)
add_subdirectory(reduction)
add_subdirectory(resample)
add_subdirectory(rmsnorm)
add_subdirectory(pointwise)
