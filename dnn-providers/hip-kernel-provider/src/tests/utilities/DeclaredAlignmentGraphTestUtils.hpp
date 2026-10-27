// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hip_kernel_provider::test_utils
{

/// Size in bytes of one element of a byte-addressable data type. Sub-byte types have no
/// element size that an alignment can be expressed in, so they are rejected.
inline int64_t elementSizeBytes(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    if(type == DataType::DOUBLE || type == DataType::INT64)
    {
        return 8;
    }
    if(type == DataType::FLOAT || type == DataType::INT32)
    {
        return 4;
    }
    if(type == DataType::HALF || type == DataType::BFLOAT16)
    {
        return 2;
    }
    if(type == DataType::UINT8 || type == DataType::INT8 || type == DataType::BOOLEAN
       || type == DataType::FP8_E4M3 || type == DataType::FP8_E5M2 || type == DataType::FP8_E8M0
       || type == DataType::FP8_E4M3_FNUZ || type == DataType::FP8_E5M2_FNUZ)
    {
        return 1;
    }
    throw std::invalid_argument("elementSizeBytes: data type has no whole-byte element size");
}

/// Re-serializes a graph with every tensor's declared alignment set to
/// @p alignmentFor(tensor data type). The builders in the test SDK always declare the default
/// alignment (16 bytes), so a graph with a different alignment has to be patched after the fact.
///
/// @param alignmentFor Callable taking the tensor's `DataType` and returning the alignment in
///        bytes.
template <typename AlignmentFn>
flatbuffers::DetachedBuffer withTensorAlignments(const void* graphBuffer,
                                                 AlignmentFn&& alignmentFor)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto graphT = std::unique_ptr<GraphT>(GetGraph(graphBuffer)->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        tensor->alignment = alignmentFor(tensor->data_type);
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

/// Asserts that every tensor of the serialized graph declares the alignment @p alignmentFor
/// gives for its data type. Guards the acceptance tests against a re-serialization that
/// silently kept the default alignment, which would make them pass for the wrong reason.
template <typename AlignmentFn>
void expectDeclaredAlignments(const void* graphBuffer, AlignmentFn&& alignmentFor)
{
    const auto* graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuffer);
    for(const auto* tensor : *graph->tensors())
    {
        EXPECT_EQ(tensor->alignment(), alignmentFor(tensor->data_type()))
            << "tensor uid " << tensor->uid();
    }
}

/// Asserts that @p isApplicable accepts @p valid as built (default alignment), and still accepts
/// it when every tensor declares only its own element size as alignment (the smallest alignment
/// a typed buffer can have), and when every tensor declares 1 byte. For engines whose kernels
/// only issue scalar element accesses, the declared alignment carries no requirement beyond
/// that. The accepted baseline is what makes the later acceptances mean something: a graph
/// that is declined for another reason cannot pass.
///
/// @param isApplicable Callable taking `const IGraph&` and returning bool.
template <typename IsApplicableFn>
void expectAcceptedAtMinimalAlignment(const flatbuffers::FlatBufferBuilder& valid,
                                      IsApplicableFn&& isApplicable)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    const auto elementSize = [](DataType type) { return elementSizeBytes(type); };
    const auto oneByte = [](DataType /*type*/) { return int64_t{1}; };

    {
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper baseline(
            valid.GetBufferPointer(), valid.GetSize());
        EXPECT_TRUE(isApplicable(baseline)) << "the graph at default alignment must be accepted";
    }
    {
        const auto aligned = withTensorAlignments(valid.GetBufferPointer(), elementSize);
        expectDeclaredAlignments(aligned.data(), elementSize);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(aligned.data(),
                                                                               aligned.size());
        EXPECT_TRUE(isApplicable(graph)) << "element-size alignment was declined";
    }
    {
        const auto aligned = withTensorAlignments(valid.GetBufferPointer(), oneByte);
        expectDeclaredAlignments(aligned.data(), oneByte);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(aligned.data(),
                                                                               aligned.size());
        EXPECT_TRUE(isApplicable(graph)) << "1-byte alignment was declined";
    }
}

} // namespace hip_kernel_provider::test_utils
