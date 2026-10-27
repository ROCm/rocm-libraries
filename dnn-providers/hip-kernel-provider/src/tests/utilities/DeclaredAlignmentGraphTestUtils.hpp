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
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

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

/// Re-serializes a graph with the declared alignment of the tensor @p uid set to @p alignment and
/// every other tensor left as built.
inline flatbuffers::DetachedBuffer
    withTensorAlignment(const void* graphBuffer, int64_t uid, int64_t alignment)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    const auto graphT = std::unique_ptr<GraphT>(GetGraph(graphBuffer)->UnPack());
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

/// Asserts how @p isApplicable follows the declared alignment of a graph whose kernels only
/// access whole elements, so that each device pointer needs to be a multiple of its element
/// size and nothing more:
///  - @p valid as built (default alignment) is accepted. This baseline is what makes the later
///    results mean something: a graph declined for another reason cannot pass.
///  - Every tensor declared at exactly its element size is accepted (the boundary).
///  - For each tensor with a device buffer, a declaration that is a multiple of the element size
///    is accepted, and one below it or one that is not a multiple is declined.
///  - Virtual tensors and pass-by-value scalars have no device pointer the backend checks, so
///    declaring 1 byte for them is accepted.
///  - Declaring 1 byte for every tensor is declined.
///
/// @param isApplicable Callable taking `const IGraph&` and returning bool.
template <typename IsApplicableFn>
void expectAcceptedAtMinimalAlignment(const flatbuffers::FlatBufferBuilder& valid,
                                      IsApplicableFn&& isApplicable)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    const auto accepts = [&isApplicable](const flatbuffers::DetachedBuffer& buffer) {
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(buffer.data(),
                                                                               buffer.size());
        return isApplicable(graph);
    };
    const auto elementSize = [](DataType type) { return elementSizeBytes(type); };
    const auto oneByte = [](DataType /*type*/) { return int64_t{1}; };
    const void* validBuffer = valid.GetBufferPointer();

    {
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper baseline(validBuffer,
                                                                                  valid.GetSize());
        EXPECT_TRUE(isApplicable(baseline)) << "the graph at default alignment must be accepted";
    }
    {
        const auto aligned = withTensorAlignments(validBuffer, elementSize);
        expectDeclaredAlignments(aligned.data(), elementSize);
        EXPECT_TRUE(accepts(aligned)) << "every tensor declared at its element size was declined";
    }

    bool anyMultiByteElement = false;
    const auto* graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(validBuffer);
    for(const auto* tensor : *graph->tensors())
    {
        const int64_t uid = tensor->uid();
        const int64_t bytes = elementSizeBytes(tensor->data_type());

        if(tensor->virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor))
        {
            EXPECT_TRUE(accepts(withTensorAlignment(validBuffer, uid, 1)))
                << "tensor uid " << uid << " has no device pointer, its alignment must not matter";
            continue;
        }

        EXPECT_TRUE(accepts(withTensorAlignment(validBuffer, uid, 2 * bytes)))
            << "tensor uid " << uid << " declared at twice its element size was declined";
        if(bytes > 1)
        {
            anyMultiByteElement = true;
            EXPECT_FALSE(accepts(withTensorAlignment(validBuffer, uid, bytes / 2)))
                << "tensor uid " << uid << " declared below its element size was accepted";
            EXPECT_FALSE(accepts(withTensorAlignment(validBuffer, uid, bytes + bytes / 2)))
                << "tensor uid " << uid << " declared at a non-multiple of its element size"
                << " was accepted";
        }
    }

    ASSERT_TRUE(anyMultiByteElement) << "the graph needs a tensor with elements wider than 1 byte";
    {
        const auto aligned = withTensorAlignments(validBuffer, oneByte);
        expectDeclaredAlignments(aligned.data(), oneByte);
        EXPECT_FALSE(accepts(aligned)) << "every tensor declared at 1 byte was accepted";
    }
}

} // namespace hip_kernel_provider::test_utils
