# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_always_built_test_sources(
    IntegrationGpuLayernormBackward.cpp
)

add_cpp_graph_test_sources(
    IntegrationGpuLayernormForward.cpp
)
