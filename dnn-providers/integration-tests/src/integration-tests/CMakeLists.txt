# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(${INTEGRATION_TESTS_EXE}
    PRIVATE
        IntegrationIsSupportedExtPerformance.cpp
)

add_subdirectory(batchnorm)
add_subdirectory(conv)
add_subdirectory(layernorm)
add_subdirectory(matmul)
add_subdirectory(reduction)
add_subdirectory(rmsnorm)
add_subdirectory(sdpa)
add_subdirectory(pointwise)
