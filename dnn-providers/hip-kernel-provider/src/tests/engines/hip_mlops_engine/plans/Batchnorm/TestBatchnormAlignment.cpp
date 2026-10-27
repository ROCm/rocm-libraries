// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "BatchnormAlignmentTestUtils.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormAlignment.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

namespace hip_kernel_provider::batchnorm::test
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

namespace
{

const TensorAttributes& tensorNamed(const GraphWrapper& graph, const std::string& name)
{
    for(const auto& entry : graph.getTensorMap())
    {
        if(entry.second->name() != nullptr && entry.second->name()->str() == name)
        {
            return *entry.second;
        }
    }
    throw std::invalid_argument("no tensor named " + name);
}

// The widest vector size the declared alignment of "x" allows, in an inference graph whose x has
// the given data type.
size_t maxVectorSizeOfX(DataType type, int64_t alignment)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, type);
    const auto patched = withNamedAlignments(builder.GetBufferPointer(), {{"x", alignment}});
    const GraphWrapper graph(patched.data(), patched.size());
    return maxVectorSizeForAlignment(tensorNamed(graph, "x"));
}

} // namespace

// Bug caught: a wrong element size, which scales every need, or an unsupported type claiming a
// size and so being covered.
TEST(TestBatchnormAlignment, DataTypeBytesCoversTheInstantiatedTypes)
{
    EXPECT_EQ(dataTypeBytes(DataType::FLOAT), 4U);
    EXPECT_EQ(dataTypeBytes(DataType::HALF), 2U);
    EXPECT_EQ(dataTypeBytes(DataType::BFLOAT16), 2U);
    EXPECT_EQ(dataTypeBytes(DataType::INT32), 0U);
    EXPECT_EQ(dataTypeBytes(DataType::DOUBLE), 0U);
}

// Bug caught: a non-positive declaration read as a usable alignment (a negative one wrapping to a
// huge size_t), or the field not read at all.
TEST(TestBatchnormAlignment, DeclaredAlignmentBytesReadsTheField)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    {
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_EQ(declaredAlignmentBytes(tensorNamed(graph, "x")), 16U);
    }

    const std::vector<std::pair<int64_t, size_t>> cases{{8, 8}, {1, 1}, {0, 0}, {-4, 0}};
    for(const auto& [declared, expected] : cases)
    {
        const auto patched = withNamedAlignments(builder.GetBufferPointer(), {{"x", declared}});
        const GraphWrapper graph(patched.data(), patched.size());
        EXPECT_EQ(declaredAlignmentBytes(tensorNamed(graph, "x")), expected)
            << "declared " << declared;
    }
}

// Bug caught: the group guarantee taken as the smallest declaration, so tensors declaring 16 and 24
// look 16-byte aligned although a pointer at address 24 is not.
TEST(TestBatchnormAlignment, CommonDeclaredAlignmentIsTheGreatestCommonDivisor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const auto patched
        = withNamedAlignments(builder.GetBufferPointer(), {{"x", 16}, {"y", 24}, {"scale", 0}});
    const GraphWrapper graph(patched.data(), patched.size());
    const auto& x = tensorNamed(graph, "x");
    const auto& y = tensorNamed(graph, "y");
    const auto& scale = tensorNamed(graph, "scale");
    const auto& bias = tensorNamed(graph, "bias");

    EXPECT_EQ(commonDeclaredAlignmentBytes({&x, &y}), 8U);
    EXPECT_FALSE(alignmentCovers(commonDeclaredAlignmentBytes({&x, &y}), 16));

    // A tensor that guarantees nothing guarantees nothing for the group.
    EXPECT_EQ(commonDeclaredAlignmentBytes({&x, &scale}), 0U);

    // Null tensors are skipped, and no tensor at all leaves the default.
    EXPECT_EQ(commonDeclaredAlignmentBytes({&bias, nullptr}), 16U);
    EXPECT_EQ(commonDeclaredAlignmentBytes({nullptr}), DEFAULT_TENSOR_ALIGNMENT_BYTES);
}

// Bug caught: the vector width not scaled by the element size, or declined at exactly the
// alignment it needs (off by one on the boundary).
TEST(TestBatchnormAlignment, MaxVectorSizeFollowsTheElementSize)
{
    // 4-byte elements: widths of 16, 8 and 4 bytes.
    EXPECT_EQ(maxVectorSizeOfX(DataType::FLOAT, 16), 4U);
    EXPECT_EQ(maxVectorSizeOfX(DataType::FLOAT, 32), 4U);
    EXPECT_EQ(maxVectorSizeOfX(DataType::FLOAT, 8), 2U);
    EXPECT_EQ(maxVectorSizeOfX(DataType::FLOAT, 24), 2U);
    EXPECT_EQ(maxVectorSizeOfX(DataType::FLOAT, 4), 1U);
    EXPECT_EQ(maxVectorSizeOfX(DataType::FLOAT, 2), 0U);

    // 2-byte elements: widths of 8, 4 and 2 bytes.
    EXPECT_EQ(maxVectorSizeOfX(DataType::HALF, 8), 4U);
    EXPECT_EQ(maxVectorSizeOfX(DataType::HALF, 4), 2U);
    EXPECT_EQ(maxVectorSizeOfX(DataType::BFLOAT16, 2), 1U);
    EXPECT_EQ(maxVectorSizeOfX(DataType::BFLOAT16, 1), 0U);
}

// Bug caught: the per-channel tensors capping the vector size in NCHW, where the kernel loads one
// element of them, or not capping it in NHWC, where it loads a vector.
TEST(TestBatchnormAlignment, InferenceVectorSizeBoundsPerChannelTensorsOnlyInNhwc)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const auto patched = withNamedAlignments(builder.GetBufferPointer(), {{"est_mean", 8}});
    const GraphWrapper graph(patched.data(), patched.size());
    const auto& x = tensorNamed(graph, "x");
    const auto& y = tensorNamed(graph, "y");
    const auto& scale = tensorNamed(graph, "scale");
    const auto& mean = tensorNamed(graph, "est_mean");

    EXPECT_EQ(maxInferenceVectorSize(true, x, y, {&scale, &mean}), 2U);
    EXPECT_EQ(maxInferenceVectorSize(false, x, y, {&scale, &mean}), 4U);
}

// Bug caught: the output tensor left out of the bound, so a y that declares less than x is read as
// if it were as aligned.
TEST(TestBatchnormAlignment, InferenceVectorSizeBoundsTheOutputInBothLayouts)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const auto patched = withNamedAlignments(builder.GetBufferPointer(), {{"y", 8}});
    const GraphWrapper graph(patched.data(), patched.size());
    const auto& x = tensorNamed(graph, "x");
    const auto& y = tensorNamed(graph, "y");
    const auto& scale = tensorNamed(graph, "scale");

    EXPECT_EQ(maxInferenceVectorSize(true, x, y, {&scale}), 2U);
    EXPECT_EQ(maxInferenceVectorSize(false, x, y, {&scale}), 2U);
}

// Bug caught: a tensor declaring less than its element size left unreported, so the kernel is
// compiled anyway.
TEST(TestBatchnormAlignment, InferenceVectorSizeIsZeroBelowTheElementAlignment)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const auto patched = withNamedAlignments(builder.GetBufferPointer(), {{"est_mean", 2}});
    const GraphWrapper graph(patched.data(), patched.size());
    const auto& x = tensorNamed(graph, "x");
    const auto& y = tensorNamed(graph, "y");
    const auto& mean = tensorNamed(graph, "est_mean");

    EXPECT_EQ(maxInferenceVectorSize(true, x, y, {&mean}), 0U);
    EXPECT_EQ(maxInferenceVectorSize(false, x, y, {&mean}), 0U);
}

} // namespace hip_kernel_provider::batchnorm::test
