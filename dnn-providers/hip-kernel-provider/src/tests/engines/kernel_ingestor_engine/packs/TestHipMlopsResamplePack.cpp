// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENGINE_HIP_MLOPS_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Container.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/HipMlopsTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestHipMlopsResamplePack.cpp
 * @brief The HIP MLOps engine on the kernel ingestor, Resample family (ALMIOPEN-3075): the
 *        engine takes over HIP_MLOPS_ENGINE's id, accepts what the hand-written builders
 *        accept and declines override shapes, and computes forward and backward resampling
 *        through the engine as CPU references do.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::hip_mlops;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr int64_t SOURCE_UID = 1; // x (forward) or dy (backward)
constexpr int64_t RESULT_UID = 2; // y (forward) or dx (backward)
constexpr int64_t INDEX_UID = 3;

// NCHW [2, 3, 6, 7], 3x3 window, stride 2, padding 1 on each side: output [2, 3, 3, 4].
constexpr int64_t N = 2;
constexpr int64_t C = 3;
constexpr int64_t IN_H = 6;
constexpr int64_t IN_W = 7;
constexpr int64_t OUT_H = 3;
constexpr int64_t OUT_W = 4;
constexpr int64_t WINDOW = 3;
constexpr int64_t STRIDE = 2;
constexpr int64_t PAD = 1;

const std::vector<int64_t> INPUT_DIMS = {N, C, IN_H, IN_W};
const std::vector<int64_t> OUTPUT_DIMS = {N, C, OUT_H, OUT_W};

struct ResampleGraph
{
    bool backward = false;
    data_objects::ResampleMode mode = data_objects::ResampleMode::MAXPOOL;
    data_objects::PaddingMode padding = data_objects::PaddingMode::NEG_INF_PAD;
    bool withIndex = false;
    bool overrideShape = false;
    bool sourceVirtual = false;
    data_objects::DataType computeType = data_objects::DataType::FLOAT;
    bool twoNodes = false;
};

flatbuffers::FlatBufferBuilder buildResampleGraph(const ResampleGraph& spec)
{
    flatbuffers::FlatBufferBuilder builder;
    // Forward reads the input shape and writes the output shape; backward the reverse.
    const auto& sourceDims = spec.backward ? OUTPUT_DIMS : INPUT_DIMS;
    const auto& resultDims = spec.backward ? INPUT_DIMS : OUTPUT_DIMS;
    const auto sourceStrides = packedStrides(sourceDims);
    const auto resultStrides = packedStrides(resultDims);
    const auto indexStrides = packedStrides(OUTPUT_DIMS);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 SOURCE_UID,
                                                                 nullptr,
                                                                 data_objects::DataType::FLOAT,
                                                                 &sourceStrides,
                                                                 &sourceDims,
                                                                 spec.sourceVirtual));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 RESULT_UID,
                                                                 nullptr,
                                                                 data_objects::DataType::FLOAT,
                                                                 &resultStrides,
                                                                 &resultDims,
                                                                 false));
    if(spec.withIndex)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     INDEX_UID,
                                                                     nullptr,
                                                                     data_objects::DataType::INT32,
                                                                     &indexStrides,
                                                                     &OUTPUT_DIMS,
                                                                     false));
    }

    const std::vector<int64_t> pads = {PAD, PAD};
    const std::vector<int64_t> strides = {STRIDE, STRIDE};
    const std::vector<int64_t> window = {WINDOW, WINDOW};
    const flatbuffers::Optional<int64_t> index
        = spec.withIndex ? flatbuffers::Optional<int64_t>(INDEX_UID) : flatbuffers::nullopt;

    flatbuffers::Offset<void> attributes;
    data_objects::NodeAttributes type{};
    if(spec.backward)
    {
        attributes = data_objects::CreateResampleBwdAttributesDirect(builder,
                                                                     SOURCE_UID,
                                                                     RESULT_UID,
                                                                     index,
                                                                     &pads,
                                                                     &pads,
                                                                     &strides,
                                                                     &window,
                                                                     spec.mode,
                                                                     spec.padding)
                         .Union();
        type = data_objects::NodeAttributes::ResampleBwdAttributes;
    }
    else
    {
        attributes = data_objects::CreateResampleFwdAttributesDirect(
                         builder,
                         SOURCE_UID,
                         RESULT_UID,
                         index,
                         &pads,
                         &pads,
                         &strides,
                         &window,
                         spec.mode,
                         spec.padding,
                         spec.withIndex ? flatbuffers::Optional<bool>(true) : flatbuffers::nullopt)
                         .Union();
        type = data_objects::NodeAttributes::ResampleFwdAttributes;
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder, "resample", spec.computeType, type, attributes));
    if(spec.twoNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(
            builder, "resample2", spec.computeType, type, attributes));
    }

    auto name = builder.CreateString("hip_mlops_resample");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShape);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

auto match(const ResampleGraph& spec)
{
    return matchGraph(buildResampleGraph(spec));
}

// ---------------------------------------------------------------------------
// Identity: the descriptor engine is HIP_MLOPS_ENGINE
// ---------------------------------------------------------------------------

TEST(TestHipMlopsResamplePack, TakesOverTheHandWrittenEnginesNameAndId)
{
    const auto& set = loadedSet(ENGINE_NAME);
    EXPECT_EQ(set.engine.builtinEngine, BUILTIN_NAME);
    // One pack per migrated operation: Resample and Layernorm, forward and backward.
    EXPECT_EQ(distinctPackIdCount(set), 4U);

    core::Container container;
    const auto ids = container.getEngineManager().getAllEngineIds();
    EXPECT_THAT(ids, ::testing::Contains(hipdnn_data_sdk::utilities::engineNameToId(BUILTIN_NAME)));
    EXPECT_THAT(ids,
                ::testing::Not(::testing::Contains(
                    hipdnn_data_sdk::utilities::engineNameToId(std::string(ENGINE_NAME)))));
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

TEST(TestHipMlopsResampleGraphMatch, RoutesEachDirectionToItsPack)
{
    EXPECT_EQ(boundOperation(match({})), "RESAMPLE_FWD");

    ResampleGraph withIndex;
    withIndex.withIndex = true;
    EXPECT_EQ(boundOperation(match(withIndex)), "RESAMPLE_FWD");

    ResampleGraph backward;
    backward.backward = true;
    backward.mode = data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING;
    backward.padding = data_objects::PaddingMode::ZERO_PAD;
    EXPECT_EQ(boundOperation(match(backward)), "RESAMPLE_BWD");
}

struct RefusalCase
{
    const char* name;
    ResampleGraph spec;
};

ResampleGraph edited(void (*edit)(ResampleGraph&))
{
    ResampleGraph spec;
    edit(spec);
    return spec;
}

class TestHipMlopsResampleRefusal : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestHipMlopsResampleRefusal, Declines)
{
    EXPECT_FALSE(match(GetParam().spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestHipMlopsResampleRefusal,
    ::testing::Values(
        // The deliberate difference from the hand-written builders.
        RefusalCase{"OverrideShapes", edited([](auto& s) { s.overrideShape = true; })},
        // The hand-written builders' own refusals.
        RefusalCase{"TwoNodes", edited([](auto& s) { s.twoNodes = true; })},
        RefusalCase{"HalfCompute",
                    edited([](auto& s) { s.computeType = data_objects::DataType::HALF; })},
        RefusalCase{"VirtualInput", edited([](auto& s) { s.sourceVirtual = true; })},
        RefusalCase{"IndexWithoutMaxpool", edited([](auto& s) {
                        s.withIndex = true;
                        s.mode = data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING;
                    })},
        RefusalCase{"BackwardMaxpoolWithoutIndex", edited([](auto& s) { s.backward = true; })}),
    [](const ::testing::TestParamInfo<RefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Numerics through the engine
// ---------------------------------------------------------------------------

/// Packed NCHW offset.
size_t at(int64_t n, int64_t c, int64_t h, int64_t w, int64_t height, int64_t width)
{
    return static_cast<size_t>(((n * C + c) * height + h) * width + w);
}

/// The forward reference, mirroring ResampleFwd.cpp's semantics: maxpool skips padding,
/// AVGPOOL_EXCLUDE_PADDING divides by the in-bounds count, INCLUDE_PADDING by the window.
std::vector<float> forwardReference(const std::vector<float>& x, data_objects::ResampleMode mode)
{
    std::vector<float> y(elementCount(OUTPUT_DIMS));
    for(int64_t n = 0; n < N; ++n)
    {
        for(int64_t c = 0; c < C; ++c)
        {
            for(int64_t oh = 0; oh < OUT_H; ++oh)
            {
                for(int64_t ow = 0; ow < OUT_W; ++ow)
                {
                    float sum = 0.0f;
                    float best = -std::numeric_limits<float>::max();
                    int64_t valid = 0;
                    for(int64_t kh = 0; kh < WINDOW; ++kh)
                    {
                        for(int64_t kw = 0; kw < WINDOW; ++kw)
                        {
                            const int64_t ih = oh * STRIDE + kh - PAD;
                            const int64_t iw = ow * STRIDE + kw - PAD;
                            if(ih < 0 || ih >= IN_H || iw < 0 || iw >= IN_W)
                            {
                                continue;
                            }
                            const float value = x[at(n, c, ih, iw, IN_H, IN_W)];
                            best = std::max(best, value);
                            sum += value;
                            ++valid;
                        }
                    }
                    float result = best;
                    if(mode == data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING)
                    {
                        result = sum / static_cast<float>(valid);
                    }
                    else if(mode == data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING)
                    {
                        result = sum / static_cast<float>(WINDOW * WINDOW);
                    }
                    y[at(n, c, oh, ow, OUT_H, OUT_W)] = result;
                }
            }
        }
    }
    return y;
}

/// The backward reference: each output window scatters dy back onto the inputs it covered,
/// divided as the forward divided (average), or onto the one input its index names (max).
std::vector<float> backwardReference(const std::vector<float>& dy,
                                     data_objects::ResampleMode mode,
                                     const std::vector<int32_t>& index)
{
    std::vector<float> dx(elementCount(INPUT_DIMS), 0.0f);
    for(int64_t n = 0; n < N; ++n)
    {
        for(int64_t c = 0; c < C; ++c)
        {
            for(int64_t oh = 0; oh < OUT_H; ++oh)
            {
                for(int64_t ow = 0; ow < OUT_W; ++ow)
                {
                    const auto o = at(n, c, oh, ow, OUT_H, OUT_W);
                    if(mode == data_objects::ResampleMode::MAXPOOL)
                    {
                        const int64_t flat = index[o];
                        dx[at(n, c, flat / IN_W, flat % IN_W, IN_H, IN_W)] += dy[o];
                        continue;
                    }
                    int64_t valid = 0;
                    for(int64_t k = 0; k < WINDOW * WINDOW; ++k)
                    {
                        const int64_t ih = oh * STRIDE + k / WINDOW - PAD;
                        const int64_t iw = ow * STRIDE + k % WINDOW - PAD;
                        valid += (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) ? 1 : 0;
                    }
                    const float divisor
                        = mode == data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING
                              ? static_cast<float>(valid)
                              : static_cast<float>(WINDOW * WINDOW);
                    for(int64_t k = 0; k < WINDOW * WINDOW; ++k)
                    {
                        const int64_t ih = oh * STRIDE + k / WINDOW - PAD;
                        const int64_t iw = ow * STRIDE + k % WINDOW - PAD;
                        if(ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W)
                        {
                            dx[at(n, c, ih, iw, IN_H, IN_W)] += dy[o] / divisor;
                        }
                    }
                }
            }
        }
    }
    return dx;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "element " << i;
    }
}

TEST(TestHipMlopsResampleDevice, ForwardMatchesTheReferenceForEveryMode)
{
    SKIP_IF_NO_DEVICES();

    const auto x = deterministicValues(elementCount(INPUT_DIMS), 0.3f);
    for(const auto mode : {data_objects::ResampleMode::MAXPOOL,
                           data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                           data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        SCOPED_TRACE(data_objects::EnumNameResampleMode(mode));
        ResampleGraph spec;
        spec.mode = mode;
        spec.padding = mode == data_objects::ResampleMode::MAXPOOL
                           ? data_objects::PaddingMode::NEG_INF_PAD
                           : data_objects::PaddingMode::ZERO_PAD;

        const DeviceBuffer<float> dx(elementCount(INPUT_DIMS), x);
        const DeviceBuffer<float> dy(elementCount(OUTPUT_DIMS));
        ASSERT_NO_FATAL_FAILURE(executeThroughTheEngine(
            buildResampleGraph(spec), {{SOURCE_UID, dx.get()}, {RESULT_UID, dy.get()}}));

        expectNear(dy.read(), forwardReference(x, mode));
    }
}

TEST(TestHipMlopsResampleDevice, MaxpoolIndexAndBackwardMatchTheReference)
{
    SKIP_IF_NO_DEVICES();

    // Forward with an index: every index names the input that holds the maximum.
    const auto x = deterministicValues(elementCount(INPUT_DIMS), 0.9f);
    ResampleGraph forward;
    forward.withIndex = true;
    const DeviceBuffer<float> xBuffer(elementCount(INPUT_DIMS), x);
    const DeviceBuffer<float> yBuffer(elementCount(OUTPUT_DIMS));
    const DeviceBuffer<int32_t> indexBuffer(elementCount(OUTPUT_DIMS));
    ASSERT_NO_FATAL_FAILURE(executeThroughTheEngine(buildResampleGraph(forward),
                                                    {{SOURCE_UID, xBuffer.get()},
                                                     {RESULT_UID, yBuffer.get()},
                                                     {INDEX_UID, indexBuffer.get()}}));
    const auto y = yBuffer.read();
    const auto index = indexBuffer.read();
    expectNear(y, forwardReference(x, data_objects::ResampleMode::MAXPOOL));
    for(int64_t n = 0; n < N; ++n)
    {
        for(int64_t c = 0; c < C; ++c)
        {
            for(int64_t o = 0; o < OUT_H * OUT_W; ++o)
            {
                const auto out = at(n, c, o / OUT_W, o % OUT_W, OUT_H, OUT_W);
                const int32_t flat = index[out];
                ASSERT_GE(flat, 0);
                ASSERT_LT(flat, IN_H * IN_W);
                EXPECT_EQ(x[at(n, c, flat / IN_W, flat % IN_W, IN_H, IN_W)], y[out]);
            }
        }
    }

    // Backward, through that index.
    const auto dyValues = deterministicValues(elementCount(OUTPUT_DIMS), 1.7f);
    ResampleGraph backward;
    backward.backward = true;
    backward.withIndex = true;
    const DeviceBuffer<float> dy(elementCount(OUTPUT_DIMS), dyValues);
    const DeviceBuffer<float> dx(elementCount(INPUT_DIMS));
    ASSERT_NO_FATAL_FAILURE(executeThroughTheEngine(
        buildResampleGraph(backward),
        {{SOURCE_UID, dy.get()}, {RESULT_UID, dx.get()}, {INDEX_UID, indexBuffer.get()}}));
    expectNear(dx.read(), backwardReference(dyValues, data_objects::ResampleMode::MAXPOOL, index));
}

TEST(TestHipMlopsResampleDevice, AveragingBackwardMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();

    const auto dyValues = deterministicValues(elementCount(OUTPUT_DIMS), 2.3f);
    for(const auto mode : {data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING,
                           data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        SCOPED_TRACE(data_objects::EnumNameResampleMode(mode));
        ResampleGraph spec;
        spec.backward = true;
        spec.mode = mode;
        spec.padding = data_objects::PaddingMode::ZERO_PAD;

        const DeviceBuffer<float> dy(elementCount(OUTPUT_DIMS), dyValues);
        const DeviceBuffer<float> dx(elementCount(INPUT_DIMS));
        ASSERT_NO_FATAL_FAILURE(executeThroughTheEngine(
            buildResampleGraph(spec), {{SOURCE_UID, dy.get()}, {RESULT_UID, dx.get()}}));

        expectNear(dx.read(), backwardReference(dyValues, mode, {}));
    }
}

} // namespace

#endif // HIPDNN_ENGINE_HIP_MLOPS_INGESTOR
