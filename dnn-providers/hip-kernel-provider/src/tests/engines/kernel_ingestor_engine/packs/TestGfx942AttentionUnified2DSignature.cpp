// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified2DTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/UnifiedAttention2DSignatureCases.hpp"

/**
 * @file TestGfx942AttentionUnified2DSignature.cpp
 * @brief Pins the kernarg ABI hipkernel:Gfx942AttentionUnified2D marshals against the ABI the
 *        rocKE builder declares. The cases are UnifiedAttention2DSignatureCases.hpp's, shared
 *        with the other unified 2D pack and run here against this pack's own prepare().
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using hipdnn_plugin_sdk::ingestor::KernelArgument;

struct SignatureArch
{
    static std::vector<KernelArgument> pythonAbi()
    {
        return gfx942_unified2d::pythonAbi();
    }

    /// Runs the pack's prepare() over the default graph and a descriptor recording
    /// @p recorded, and returns the message of whatever it throws. Empty means it returned a
    /// prepared dispatch, which no case expects.
    static std::string prepareFailureFor(const std::vector<KernelArgument>& recorded)
    {
        using namespace gfx942_unified2d;
        const Graph graph;
        const auto bound = matchGraph(graph);
        EXPECT_TRUE(bound.has_value()) << "graph_match declined the fixture graph, so prepare() "
                                          "never ran and no case below means anything";
        if(!bound.has_value())
        {
            return {};
        }
        return prepareFailure(graph, *bound, makeKpackKernel(KernelSpec{}, recorded));
    }
};

} // namespace

namespace signature = unified2d_signature;

TEST(TestGfx942AttentionUnified2DSignature, AcceptsTheAbiThePythonDeclares)
{
    signature::acceptsTheAbiThePythonDeclares<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesADroppedTrailingArgument)
{
    signature::refusesADroppedTrailingArgument<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesAnExtraTrailingArgument)
{
    signature::refusesAnExtraTrailingArgument<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesTheFormWithoutTheTrailingStrides)
{
    signature::refusesTheFormWithoutTheTrailingStrides<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesAScalarInAPointerSlot)
{
    signature::refusesAScalarInAPointerSlot<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesAPointerInAScalarSlot)
{
    signature::refusesAPointerInAScalarSlot<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesBuffersAndScalarsTransposed)
{
    signature::refusesBuffersAndScalarsTransposed<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesANarrowedPointerSlot)
{
    signature::refusesANarrowedPointerSlot<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesAWidenedScalarSlot)
{
    signature::refusesAWidenedScalarSlot<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesEachSlotUnderAnotherName)
{
    signature::refusesEachSlotUnderAnotherName<SignatureArch>();
}

TEST(TestGfx942AttentionUnified2DSignature, RefusesEachExchangeOfTwoLikeSlots)
{
    signature::refusesEachExchangeOfTwoLikeSlots<SignatureArch>();
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
