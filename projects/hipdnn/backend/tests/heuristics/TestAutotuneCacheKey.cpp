// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestAutotuneCacheKey.cpp
 * @brief Contract tests for `deriveCacheKey`, backed by the real `GraphContentKey`. The
 *        graph view must be a verifiable FlatBuffer `Graph` buffer -- `GraphWrapper`
 *        declines anything else exactly like an empty view.
 */

#include "heuristics/config/AutotuneCacheKey.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace hipdnn_backend::heuristics::config;

namespace
{
namespace fb = hipdnn_flatbuffers_sdk::data_objects;

hipdnnPluginConstData_t asConstData(const std::array<uint8_t, 4>& bytes)
{
    return {bytes.data(), bytes.size()};
}

/// Builds a minimal verifiable `Graph` FlatBuffer with a single tensor of the given uid.
/// A tensor uid's ordinal folds into the hash; its value does not, so two graphs
/// differing only in uid value key identically.
std::vector<uint8_t> buildSingleTensorGraphBuffer(int64_t uid, int64_t dim)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims{dim};
    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(builder, uid, "t", fb::DataType::FLOAT, nullptr, &dims)};
    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             nullptr,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}
} // namespace

TEST(TestAutotuneCacheKey, EmptyGraphViewReturnsNullopt)
{
    const std::array<uint8_t, 4> device{1, 2, 3, 4};
    const hipdnnPluginConstData_t emptyGraph{nullptr, 0};

    EXPECT_FALSE(deriveCacheKey(emptyGraph, asConstData(device)).has_value());
}

TEST(TestAutotuneCacheKey, EmptyDeviceViewReturnsNullopt)
{
    const std::array<uint8_t, 4> graph{1, 2, 3, 4};
    const hipdnnPluginConstData_t emptyDevice{nullptr, 0};

    EXPECT_FALSE(deriveCacheKey(asConstData(graph), emptyDevice).has_value());
}

TEST(TestAutotuneCacheKey, UnverifiableGraphViewReturnsNullopt)
{
    const std::array<uint8_t, 4> notAGraph{1, 2, 3, 4};
    const std::array<uint8_t, 4> device{5, 6, 7, 8};

    EXPECT_FALSE(deriveCacheKey(asConstData(notAGraph), asConstData(device)).has_value());
}

TEST(TestAutotuneCacheKey, IdenticalBytesHashIdentically)
{
    const auto graph = buildSingleTensorGraphBuffer(1, 4);
    const std::array<uint8_t, 4> device{5, 6, 7, 8};
    const hipdnnPluginConstData_t graphView{graph.data(), graph.size()};

    const auto first = deriveCacheKey(graphView, asConstData(device));
    const auto second = deriveCacheKey(graphView, asConstData(device));

    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*first, *second);
}

TEST(TestAutotuneCacheKey, DifferingBytesHashDifferently)
{
    const auto graphA = buildSingleTensorGraphBuffer(1, 4);
    const auto graphB = buildSingleTensorGraphBuffer(1, 8);
    const std::array<uint8_t, 4> device{5, 6, 7, 8};
    const hipdnnPluginConstData_t graphViewA{graphA.data(), graphA.size()};
    const hipdnnPluginConstData_t graphViewB{graphB.data(), graphB.size()};

    const auto keyA = deriveCacheKey(graphViewA, asConstData(device));
    const auto keyB = deriveCacheKey(graphViewB, asConstData(device));

    ASSERT_TRUE(keyA.has_value());
    ASSERT_TRUE(keyB.has_value());
    EXPECT_NE(*keyA, *keyB);
}

TEST(TestAutotuneCacheKey, RenumberedGraphHashesIdentically)
{
    // A tensor's ordinal folds into the key, not its uid value, so a renumbered but
    // otherwise identical graph keys the same.
    const auto graphA = buildSingleTensorGraphBuffer(1, 4);
    const auto graphB = buildSingleTensorGraphBuffer(42, 4);
    const std::array<uint8_t, 4> device{5, 6, 7, 8};
    const hipdnnPluginConstData_t graphViewA{graphA.data(), graphA.size()};
    const hipdnnPluginConstData_t graphViewB{graphB.data(), graphB.size()};

    const auto keyA = deriveCacheKey(graphViewA, asConstData(device));
    const auto keyB = deriveCacheKey(graphViewB, asConstData(device));

    ASSERT_TRUE(keyA.has_value());
    ASSERT_TRUE(keyB.has_value());
    EXPECT_EQ(*keyA, *keyB);
}
