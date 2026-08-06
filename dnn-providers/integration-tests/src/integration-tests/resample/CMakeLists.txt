# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Resample has no bundle equivalent yet, so this stays built with
# BUILD_CPP_GRAPH_TESTS=OFF to avoid dropping the op's only coverage. It is
# listed in HIPDNN_IT_ALWAYS_BUILT_SOURCES as migration debt — move it to
# add_cpp_graph_test_sources() once resample bundles land.
add_always_built_test_sources(
    IntegrationGpuResampleForward.cpp
)
