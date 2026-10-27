// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hip_kernel_provider::test_utils
{

/// Re-serializes a graph with the declared alignment (bytes) of the tensor @p uid replaced.
inline flatbuffers::DetachedBuffer withDeclaredAlignment(
    const flatbuffers::FlatBufferBuilder& graph, int64_t uid, int64_t alignment)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto graphT = std::unique_ptr<GraphT>(GetGraph(graph.GetBufferPointer())->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        if(tensor->uid == uid)
        {
            tensor->alignment = alignment;
        }
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

/// One tensor of a graph and the alignment (bytes) its kernel needs from the declared value.
struct AlignmentRole
{
    const char* name;
    int64_t uid;
    int64_t needBytes;
};

/// Asserts that @p isApplicable accepts @p valid, and that for each role the declared alignment
/// of that one tensor decides the outcome the way a multiple guarantee must:
///   - declared == need is accepted (boundary),
///   - declared == 2 x need is accepted (a larger multiple still guarantees the need),
///   - declared == need / 2, need x 3 / 2 (not a multiple of need), 1 and 0 are declined.
/// The accepted baseline is what makes the declines mean something: a check that refuses every
/// graph cannot pass.
///
/// @param isApplicable Callable taking `const IGraph&` and returning bool.
template <typename IsApplicableFn>
void expectDeclaredAlignmentHonored(const flatbuffers::FlatBufferBuilder& valid,
                                    const std::vector<AlignmentRole>& roles,
                                    IsApplicableFn&& isApplicable)
{
    {
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper baseline(
            valid.GetBufferPointer(), valid.GetSize());
        EXPECT_TRUE(isApplicable(baseline)) << "the graph with default alignment must be accepted";
    }

    const auto accepts = [&](int64_t uid, int64_t alignment) {
        const auto buffer = withDeclaredAlignment(valid, uid, alignment);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(buffer.data(),
                                                                               buffer.size());
        return isApplicable(graph);
    };

    for(const auto& role : roles)
    {
        EXPECT_TRUE(accepts(role.uid, role.needBytes))
            << role.name << " declaring exactly its need (" << role.needBytes
            << " bytes) was declined";
        EXPECT_TRUE(accepts(role.uid, role.needBytes * 2))
            << role.name << " declaring twice its need was declined";
        EXPECT_FALSE(accepts(role.uid, role.needBytes / 2))
            << role.name << " declaring half its need (" << role.needBytes / 2
            << " bytes) was accepted";
        EXPECT_FALSE(accepts(role.uid, role.needBytes + role.needBytes / 2))
            << role.name << " declaring a non-multiple of its need ("
            << role.needBytes + role.needBytes / 2 << " bytes) was accepted";
        EXPECT_FALSE(accepts(role.uid, 1)) << role.name << " declaring 1 byte was accepted";
        EXPECT_FALSE(accepts(role.uid, 0)) << role.name << " declaring 0 bytes was accepted";
    }
}

} // namespace hip_kernel_provider::test_utils
