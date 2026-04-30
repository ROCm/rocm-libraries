// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <gtest/gtest.h>

#include "HipKernelUtils.hpp"

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider::hip_kernel_utils;

namespace
{

/// Creates a minimal flatbuffer graph containing a single tensor with the given dims and strides.
/// Returns the FlatBufferBuilder that owns the buffer.
flatbuffers::FlatBufferBuilder createSingleTensorGraph(const std::vector<int64_t>& dims,
                                                       const std::vector<int64_t>& strides)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "tensor",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

/// Extracts the TensorAttributes pointer (UID 1) from a single-tensor graph.
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    getTensor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& graph)
{
    return graph.getTensorMap().at(1);
}

} // namespace

// ============================================================================
// findDeviceBuffer
// ============================================================================

TEST(TestFindDeviceBuffer, FindsBufferWithMatchingUid)
{
    int data1 = 0;
    int data2 = 0;
    std::array<hipdnnPluginDeviceBuffer_t, 2> buffers = {{{1, &data1}, {2, &data2}}};

    auto result = findDeviceBuffer(2, buffers.data(), 2);

    EXPECT_EQ(result.uid, 2);
    EXPECT_EQ(result.ptr, &data2);
}

TEST(TestFindDeviceBuffer, FindsFirstBufferInArray)
{
    int data = 0;
    std::array<hipdnnPluginDeviceBuffer_t, 1> buffers = {{{42, &data}}};

    auto result = findDeviceBuffer(42, buffers.data(), 1);

    EXPECT_EQ(result.uid, 42);
    EXPECT_EQ(result.ptr, &data);
}

TEST(TestFindDeviceBuffer, ThrowsWhenUidNotFound)
{
    int data = 0;
    std::array<hipdnnPluginDeviceBuffer_t, 1> buffers = {{{1, &data}}};

    EXPECT_THROW(findDeviceBuffer(99, buffers.data(), 1), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestFindDeviceBuffer, ThrowsWhenBufferArrayIsEmpty)
{
    EXPECT_THROW(findDeviceBuffer(1, nullptr, 0), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestFindDeviceBuffer, FindsFirstMatchWhenDuplicateUidsExist)
{
    int data1 = 0;
    int data2 = 0;
    std::array<hipdnnPluginDeviceBuffer_t, 2> buffers = {{{5, &data1}, {5, &data2}}};

    auto result = findDeviceBuffer(5, buffers.data(), 2);

    EXPECT_EQ(result.uid, 5);
    EXPECT_EQ(result.ptr, &data1);
}

// ============================================================================
// findTensorAttributes
// ============================================================================

TEST(TestFindTensorAttributes, FindsTensorWithMatchingUid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();

    // UID 1 is the x tensor in the test graph
    const auto& attrs = findTensorAttributes(tensorMap, 1);
    EXPECT_EQ(attrs.uid(), 1);
}

TEST(TestFindTensorAttributes, ThrowsWhenUidNotFound)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();

    EXPECT_THROW(findTensorAttributes(tensorMap, 9999), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestFindTensorAttributes, ThrowsWhenMapIsEmpty)
{
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>
        emptyMap;

    EXPECT_THROW(findTensorAttributes(emptyMap, 1), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// isChannelLastLayout - 4D
// ============================================================================

TEST(TestIsChannelLastLayout, ReturnsTrueForNhwc4D)
{
    std::vector<int64_t> dims = {1, 3, 224, 224};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(
        dims, hipdnn_data_sdk::utilities::TensorLayout::NHWC.strideOrder);

    auto builder = createSingleTensorGraph(dims, strides);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_TRUE(isChannelLastLayout(getTensor(graph)));
}

TEST(TestIsChannelLastLayout, ReturnsFalseForNchw4D)
{
    std::vector<int64_t> dims = {1, 3, 224, 224};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(
        dims, hipdnn_data_sdk::utilities::TensorLayout::NCHW.strideOrder);

    auto builder = createSingleTensorGraph(dims, strides);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_FALSE(isChannelLastLayout(getTensor(graph)));
}

// ============================================================================
// isChannelLastLayout - 5D
// ============================================================================

TEST(TestIsChannelLastLayout, ReturnsTrueForNdhwc5D)
{
    std::vector<int64_t> dims = {1, 3, 8, 224, 224};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(
        dims, hipdnn_data_sdk::utilities::TensorLayout::NDHWC.strideOrder);

    auto builder = createSingleTensorGraph(dims, strides);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_TRUE(isChannelLastLayout(getTensor(graph)));
}

TEST(TestIsChannelLastLayout, ReturnsFalseForNcdhw5D)
{
    std::vector<int64_t> dims = {1, 3, 8, 224, 224};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(
        dims, hipdnn_data_sdk::utilities::TensorLayout::NCDHW.strideOrder);

    auto builder = createSingleTensorGraph(dims, strides);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_FALSE(isChannelLastLayout(getTensor(graph)));
}

// ============================================================================
// isChannelLastLayout - error cases
// ============================================================================

TEST(TestIsChannelLastLayout, ThrowsFor3DTensor)
{
    std::vector<int64_t> dims = {1, 3, 224};
    std::vector<int64_t> strides = {672, 224, 1};

    auto builder = createSingleTensorGraph(dims, strides);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_THROW(isChannelLastLayout(getTensor(graph)), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestIsChannelLastLayout, ThrowsFor6DTensor)
{
    std::vector<int64_t> dims = {1, 3, 4, 8, 224, 224};
    std::vector<int64_t> strides = {4816896, 1605632, 401408, 50176, 224, 1};

    auto builder = createSingleTensorGraph(dims, strides);
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    EXPECT_THROW(isChannelLastLayout(getTensor(graph)), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// parseActivation
// ============================================================================

namespace
{

// Helper function to create PointwiseAttributes
flatbuffers::FlatBufferBuilder
    createPointwiseAttributes(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode,
                              flatbuffers::Optional<float> reluLowerClip = flatbuffers::nullopt,
                              flatbuffers::Optional<float> reluUpperClip = flatbuffers::nullopt,
                              flatbuffers::Optional<float> reluLowerClipSlope
                              = flatbuffers::nullopt,
                              flatbuffers::Optional<float> eluAlpha = flatbuffers::nullopt,
                              flatbuffers::Optional<float> softplusBeta = flatbuffers::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset
        = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                          mode,
                                                                          reluLowerClip,
                                                                          reluUpperClip,
                                                                          reluLowerClipSlope,
                                                                          flatbuffers::nullopt,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          1,
                                                                          flatbuffers::nullopt,
                                                                          eluAlpha,
                                                                          softplusBeta);
    builder.Finish(attrOffset);
    return builder;
}

} // namespace

TEST(TestParseActivation, ReluDefault)
{
    auto builder
        = createPointwiseAttributes(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);

    EXPECT_EQ(params.mode, ActivationMode::RELU);
    EXPECT_DOUBLE_EQ(params.alpha, 0.0);
}

TEST(TestParseActivation, ReluClippedUpperOnly)
{
    auto builder = createPointwiseAttributes(
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, flatbuffers::nullopt, 5.0f);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);

    EXPECT_EQ(params.mode, ActivationMode::CLIPPED_RELU);
    EXPECT_NEAR(params.alpha, 5.0f, 1e-10);
}

TEST(TestParseActivation, ReluClampLowerAndUpper)
{
    auto builder = createPointwiseAttributes(
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, 0.5f, 10.0f);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);

    EXPECT_EQ(params.mode, ActivationMode::CLAMP);
    EXPECT_NEAR(params.alpha, 0.5f, 1e-10);
    EXPECT_NEAR(params.beta, 10.0f, 1e-10);
}

TEST(TestParseActivation, ReluLeaky)
{
    auto builder
        = createPointwiseAttributes(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    0.01f);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);

    EXPECT_EQ(params.mode, ActivationMode::LEAKY_RELU);
    EXPECT_NEAR(params.alpha, 0.01f, 1e-10);
}

TEST(TestParseActivation, ThrowsOnReluUnsupportedLowerClipOnly)
{
    // lower clip without upper clip is not supported
    auto builder = createPointwiseAttributes(
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD, 0.5f, flatbuffers::nullopt);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(parseActivation(*attrs), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestParseActivation, EluCustomAlpha)
{
    auto builder
        = createPointwiseAttributes(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_FWD,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    flatbuffers::nullopt,
                                    1.50f);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);

    EXPECT_EQ(params.mode, ActivationMode::ELU);
    EXPECT_NEAR(params.alpha, 1.50f, 1e-10);
}

TEST(TestParseActivation, SoftplusValidBeta)
{
    auto builder = createPointwiseAttributes(
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1.0f);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);

    EXPECT_EQ(params.mode, ActivationMode::SOFTRELU);
}

TEST(TestParseActivation, SoftplusInvalidBeta)
{
    auto builder = createPointwiseAttributes(
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2.0f);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    EXPECT_THROW(parseActivation(*attrs), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestParseActivation, Sigmoid)
{
    auto builder = createPointwiseAttributes(
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);
    EXPECT_EQ(params.mode, ActivationMode::LOGISTIC);
}

TEST(TestParseActivation, TanhWithDefaults)
{
    auto builder
        = createPointwiseAttributes(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_BWD);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);

    EXPECT_EQ(params.mode, ActivationMode::TANH);
    EXPECT_DOUBLE_EQ(params.alpha, 1.0);
    EXPECT_DOUBLE_EQ(params.beta, 1.0);
}

TEST(TestParseActivation, Passthru)
{
    auto builder
        = createPointwiseAttributes(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY);
    const auto* attrs
        = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());

    auto params = parseActivation(*attrs);

    EXPECT_EQ(params.mode, ActivationMode::PASTHRU);
}
