// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "tests/engines/kernel_ingestor_engine/packs/ConvAotModelBindingTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdAotTestSupport.hpp"

/**
 * @file TestGfx942ConvFwdAotModelBinding.cpp
 * @brief Census of the trained kernel ranker hipkernel:Gfx942ConvFwdAot binds in its UED:
 *        it must survive the loader's pre-flight, and it must score every kernel the packed
 *        catalog ships, on a dense and a depthwise 3x3 forward convolution of each dtype.
 *        The checks, and why they fail the build, are ConvAotModelBindingTestSupport.hpp's.
 *
 * Hand-written, not generated: it belongs to the census binary beside the generated
 * TestGfx942ConvFwdAotPacks, and is registered with its own hkp_register_census_tests()
 * call in src/tests/CMakeLists.txt.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::conv_aot_model_binding_test;
namespace fwd = hip_kernel_provider::kernel_ingestor_engine::conv_fwd_aot_test;

constexpr CensusEngine GFX942_CONV_FWD_AOT{"hipkernel:Gfx942ConvFwdAot", "gfx942", 64};

/// A dense and a depthwise 3x3 forward convolution, packed NHWC, in @p dtype. Every
/// shipped kernel is scored on both, whether the graph admits it or not.
std::vector<ScoredGraph> scoredGraphs(hipdnn_flatbuffers_sdk::data_objects::DataType dtype)
{
    const auto dense = fwd::denseShape(64, 64, 28, 28);
    const auto depthwise = fwd::depthwiseShape(64, 28, 28, 3, 1, 1);
    std::vector<ScoredGraph> graphs;
    graphs.push_back(
        {"dense 3x3 (64 -> 64 channels)", fwd::buildGraph(fwd::convSpec(dense, dtype))});
    graphs.push_back(
        {"depthwise 3x3 (64 groups)", fwd::buildGraph(fwd::convSpec(depthwise, dtype))});
    return graphs;
}

TEST(TestGfx942ConvFwdAotModelBinding, TheBoundL2KernelRankerLoads)
{
    expectBoundRankerLoads(GFX942_CONV_FWD_AOT);
}

TEST(TestGfx942ConvFwdAotModelBinding, TheBoundL2KernelRankerScoresEveryShippedKernel)
{
    expectBoundRankerScoresEveryShippedKernel(GFX942_CONV_FWD_AOT, "conv fwd", scoredGraphs);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
