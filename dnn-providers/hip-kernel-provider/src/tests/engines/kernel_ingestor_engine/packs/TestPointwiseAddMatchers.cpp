// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/PointwiseAddMatchers.hpp"
#include "engines/kernel_ingestor_engine/packs/PointwiseAddSymbols.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseAddTestGraphs.hpp"

/**
 * @file TestNativeMatchers.cpp
 * @brief The pack's two matcher shapes: what each accepts, and what each refuses.
 *
 * The refusals matter more than the acceptances here. An under-specified decline accepts
 * a graph the kernel cannot serve, which is a wrong answer rather than a missed
 * optimization — and for a prebuilt kernel it is a silent one.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::utilities::parseUuid;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

hipDeviceProp_t testDeviceProperties()
{
    hipDeviceProp_t properties{};
    properties.warpSize = 64;
    return properties;
}

KernelDefinition makeKernel(int64_t blockSize, const std::string& dtype)
{
    KernelDefinition kernel;
    kernel.kernelId = parseUuid("00000000-0000-4000-8000-000000000001");
    kernel.packId = parseUuid("00000000-0000-4000-8000-000000000002");
    kernel.dispatchId = parseUuid("00000000-0000-4000-8000-000000000003");
    kernel.sourceFile = "PointwiseAdd.cpp";
    kernel.entryPoint = "PointwiseAdd";
    kernel.metadata
        = {{std::string(BLOCK_SIZE_FIELD), blockSize}, {std::string(DTYPE_FIELD), dtype}};
    return kernel;
}

/// Wraps a built graph buffer so a test reads it the way an engine does.
class GraphFixture
{
public:
    explicit GraphFixture(flatbuffers::FlatBufferBuilder builder)
        : _builder(std::move(builder))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
        , _properties(testDeviceProperties())
    {
    }

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    hipDeviceProp_t _properties;
};

// ---------------------------------------------------------------------------
// Graph-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddGraphMatcher, AcceptsASingleElementFloatAdd)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(pointwiseAddGraphMatches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, AcceptsAHalfPrecisionAdd)
{
    // The graph-level gate is dtype-agnostic within the pack's declared set; pinning the
    // kernel's baked dtype is the kernel-scoped matcher's job.
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::HALF));

    EXPECT_TRUE(pointwiseAddGraphMatches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, RefusesAnotherPointwiseOperation)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::MUL));

    EXPECT_FALSE(pointwiseAddGraphMatches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, RefusesMultiElementTensors)
{
    // The kernel writes element 0 and nothing else, so anything larger would silently
    // leave most of the output untouched.
    const GraphFixture fixture(buildPointwiseGraph(
        data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 2, 2}));

    EXPECT_FALSE(pointwiseAddGraphMatches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, RefusesARankTheDispatchPathCannotServe)
{
    // A 1-element 1-D tensor suits the kernel, which indexes element 0, but the
    // provider's compile options derive layout from the tensor and reject anything
    // below rank 4. Accepting it would trade a free decline at applicability for a
    // failed plan build, which the caller pays for.
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1}));

    EXPECT_FALSE(pointwiseAddGraphMatches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, AcceptsTheUpperSupportedRank)
{
    const GraphFixture fixture(buildPointwiseGraph(
        data_objects::PointwiseMode::ADD, data_objects::DataType::FLOAT, {1, 1, 1, 1, 1}));

    EXPECT_TRUE(pointwiseAddGraphMatches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, RefusesAUnaryPointwise)
{
    const GraphFixture fixture(buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                                   data_objects::DataType::FLOAT,
                                                   {1, 1, 1, 1},
                                                   std::nullopt,
                                                   /*binary=*/false));

    EXPECT_FALSE(pointwiseAddGraphMatches(fixture.context()));
}

TEST(TestPointwiseAddGraphMatcher, RefusesAMultiNodeGraph)
{
    // A prebuilt kernel serves one complete graph, so a larger graph is a different
    // problem even though it contains this one.
    const GraphFixture fixture(buildTwoNodePointwiseGraph());

    EXPECT_FALSE(pointwiseAddGraphMatches(fixture.context()));
}

// ---------------------------------------------------------------------------
// Kernel-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddKernelMatcher, AcceptsAKernelWhoseDtypeMatchesTheGraph)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(pointwiseAddKernelMatches(fixture.context(), makeKernel(64, "FLOAT")));
}

TEST(TestPointwiseAddKernelMatcher, RefusesAKernelBakedForAnotherDtype)
{
    // The failure this prevents is silent: an f16 binary handed f32 operands does not
    // fail, it returns wrong numbers.
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_FALSE(pointwiseAddKernelMatches(fixture.context(), makeKernel(64, "HALF")));
}

TEST(TestPointwiseAddKernelMatcher, AcceptsAHalfKernelForAHalfGraph)
{
    const GraphFixture fixture(
        buildPointwiseGraph(data_objects::PointwiseMode::ADD, data_objects::DataType::HALF));

    EXPECT_TRUE(pointwiseAddKernelMatches(fixture.context(), makeKernel(64, "HALF")));
}

TEST(TestPointwiseAddKernelMatcher, IgnoresBlockSizeWhichTheGraphDoesNotConstrain)
{
    // block_size is a ranking and launch axis, not an applicability one: every block
    // size serves every graph this pack accepts.
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_TRUE(pointwiseAddKernelMatches(fixture.context(), makeKernel(64, "FLOAT")));
    EXPECT_TRUE(pointwiseAddKernelMatches(fixture.context(), makeKernel(256, "FLOAT")));
}

// ---------------------------------------------------------------------------
// Score and binding
// ---------------------------------------------------------------------------

TEST(TestPointwiseAddScore, PrefersTheLargerBlockSize)
{
    const GraphFixture fixture(buildPointwiseGraph());

    EXPECT_GT(pointwiseAddScore(makeKernel(256, "FLOAT"), fixture.context()),
              pointwiseAddScore(makeKernel(64, "FLOAT"), fixture.context()));
}

TEST(TestPointwiseAddBinding, ReportsTheOperandUidsInArgumentOrder)
{
    // Dispatch binds its arguments from these, rather than re-deriving the graph shape
    // with a second notion of what a pointwise add looks like.
    const GraphFixture fixture(buildPointwiseGraph());

    const auto binding = pointwiseAddBinding(fixture.context());

    EXPECT_EQ(binding.inputA, INPUT_A_UID);
    EXPECT_EQ(binding.inputB, INPUT_B_UID);
    EXPECT_EQ(binding.output, OUTPUT_UID);
}

TEST(TestPointwiseAddBinding, RefusesAGraphTheMatcherWouldReject)
{
    const GraphFixture fixture(buildTwoNodePointwiseGraph());

    EXPECT_THROW(pointwiseAddBinding(fixture.context()), hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
