// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "tests/engines/kernel_ingestor_engine/packs/ConvAotModelBindingTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvDgradMatcherTestSupport.hpp"

/**
 * @file ConvDgradModelBindingTestSupport.hpp
 * @brief The ModelBinding census of a dgrad AOT engine's trained kernel ranker, shared by
 *        the per-engine suites (TestGfx942ConvDgradAotModelBinding.cpp,
 *        TestGfx1250ConvDgradAotModelBinding.cpp): ConvAotModelBindingTestSupport.hpp's
 *        checks, run on a dense and a depthwise backward-data graph of each dtype. Why they
 *        fail the build is said there.
 *
 * Each suite keeps its own TEST names, which src/tests/CMakeLists.txt registers per suite.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_model_binding_test
{

using conv_aot_model_binding_test::ARCH_KEY;
using conv_aot_model_binding_test::CensusEngine;
using conv_aot_model_binding_test::deviceOf;
using conv_aot_model_binding_test::expectBoundRankerLoads;
using conv_aot_model_binding_test::METRIC;
using conv_aot_model_binding_test::ROLE;
using conv_aot_model_binding_test::shippedKernelsByDtype;

/// What each dtype's kernels are scored on: a dense 3x3 and a depthwise 3x3 backward-data
/// convolution. Every shipped kernel is scored on both, whether the graph admits it or not.
inline std::array<dgrad_matcher_test::DgradShape, 2> scoredShapes()
{
    dgrad_matcher_test::DgradShape depthwise;
    depthwise.c = 64;
    depthwise.k = 64;
    depthwise.groups = 64;
    depthwise.h = 28;
    depthwise.w = 28;
    return {dgrad_matcher_test::DgradShape{}, depthwise};
}

/// TheBoundL2KernelRankerScoresEveryShippedKernel: the ranker @p engine's UED binds scores
/// every kernel it ships, one at a time and as one catalog, on each dtype's scoredShapes().
inline void expectBoundRankerScoresEveryShippedKernel(const CensusEngine& engine)
{
    conv_aot_model_binding_test::expectBoundRankerScoresEveryShippedKernel(
        engine, "conv dgrad", [](hipdnn_flatbuffers_sdk::data_objects::DataType dtype) {
            std::vector<conv_aot_model_binding_test::ScoredGraph> graphs;
            for(const auto& shape : scoredShapes())
            {
                graphs.push_back(
                    {"with " + std::to_string(shape.groups) + " groups",
                     dgrad_matcher_test::buildGraph(dgrad_matcher_test::graphSpec(shape, dtype))});
            }
            return graphs;
        });
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::dgrad_model_binding_test

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
