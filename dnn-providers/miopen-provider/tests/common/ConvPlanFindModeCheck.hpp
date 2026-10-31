// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <utility>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "MiopenApi.hpp"

namespace test_conv_common
{

/// Checks that a convolution plan built with benchmarking leaves its descriptor in Normal find
/// mode, as PyTorch's cudnn.benchmark=True does: MIOpen then neither returns an installed
/// find-db entry unmeasured nor skips non-dynamic solvers. A plan built without benchmarking
/// keeps MIOpen's default. Skips when that default is already Normal (MIOPEN_FIND_MODE=NORMAL),
/// where the two cases cannot be told apart.
template <typename Params, typename Plan, typename Attributes>
void expectBenchmarkingSelectsNormalFindMode(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& graph,
    const Attributes& attributes)
{
    for(const bool benchmarking : {false, true})
    {
        SCOPED_TRACE(benchmarking ? "benchmarking" : "default");
        Params params(attributes, graph.getTensorMap());
        auto* convDescriptor = params.conv().convDescriptor();
        miopenConvolutionFindMode_t defaultMode{};
        ASSERT_EQ(miopenGetConvolutionFindMode(convDescriptor, &defaultMode), miopenStatusSuccess);
        if(defaultMode == miopenConvolutionFindModeNormal)
        {
            GTEST_SKIP() << "MIOpen's default find mode is already Normal (MIOPEN_FIND_MODE)";
        }

        HipdnnMiopenSettings settings;
        settings.setBenchmarkingEnabled(benchmarking);
        const Plan plan(handle, std::move(params), settings);

        miopenConvolutionFindMode_t mode{};
        ASSERT_EQ(miopenGetConvolutionFindMode(convDescriptor, &mode), miopenStatusSuccess);
        EXPECT_EQ(mode, benchmarking ? miopenConvolutionFindModeNormal : defaultMode);
    }
}

} // namespace test_conv_common
