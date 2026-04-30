// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltUtils.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

using namespace hipblaslt_plugin;
using namespace hipblaslt_utils;
using PM = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

namespace
{

/// Helper that builds a FlatBuffer-backed PointwiseAttributes and keeps the
/// builder alive for the lifetime of the returned object.
struct PointwiseAttrsHolder
{
    flatbuffers::FlatBufferBuilder builder;
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* attrs = nullptr;

    PointwiseAttrsHolder(PM operation,
                         flatbuffers::Optional<float> reluLowerClip = flatbuffers::nullopt,
                         flatbuffers::Optional<float> reluUpperClip = flatbuffers::nullopt,
                         flatbuffers::Optional<float> swishBeta = flatbuffers::nullopt)
    {
        auto offset
            = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                              operation,
                                                                              reluLowerClip,
                                                                              reluUpperClip,
                                                                              flatbuffers::nullopt,
                                                                              flatbuffers::nullopt,
                                                                              0,
                                                                              flatbuffers::nullopt,
                                                                              flatbuffers::nullopt,
                                                                              0,
                                                                              swishBeta);
        builder.Finish(offset);
        attrs = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes>(
            builder.GetBufferPointer());
    }
};

} // anonymous namespace

TEST(TestHipblasltUtils, MapPointwiseModeToHipblasLtEpilogue)
{
    // -- Null attrs ----------------------------------------------------------
    {
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(nullptr, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_DEFAULT);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(nullptr, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_BIAS);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }

    // -- ReLU / Clamp --------------------------------------------------------
    {
        // RELU_FWD with both clips → CLAMP without bias
        PointwiseAttrsHolder h(PM::RELU_FWD, -1.0f, 6.0f);
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_CLAMP_EXT);
        EXPECT_FLOAT_EQ(result.act0, -1.0f);
        EXPECT_FLOAT_EQ(result.act1, 6.0f);
    }
    {
        // RELU_FWD with both clips → CLAMP with bias
        PointwiseAttrsHolder h(PM::RELU_FWD, -1.0f, 6.0f);
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT);
        EXPECT_FLOAT_EQ(result.act0, -1.0f);
        EXPECT_FLOAT_EQ(result.act1, 6.0f);
    }
    {
        // RELU_FWD standard (lower_clip = 0, no upper_clip) without bias
        PointwiseAttrsHolder h(PM::RELU_FWD, 0.0f);
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_RELU);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        // RELU_FWD standard (lower_clip = 0, no upper_clip) with bias
        PointwiseAttrsHolder h(PM::RELU_FWD, 0.0f);
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_RELU_BIAS);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        // RELU_FWD with non-zero lower_clip only → throws
        PointwiseAttrsHolder h(PM::RELU_FWD, 1.0f);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    {
        // RELU_FWD with no clips at all → throws
        PointwiseAttrsHolder h(PM::RELU_FWD);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }

    // -- GELU ----------------------------------------------------------------
    {
        PointwiseAttrsHolder h(PM::GELU_APPROX_TANH_FWD);
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_GELU);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        PointwiseAttrsHolder h(PM::GELU_APPROX_TANH_FWD);
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_GELU_BIAS);
    }

    // -- Swish ---------------------------------------------------------------
    {
        PointwiseAttrsHolder h(PM::SWISH_FWD, flatbuffers::nullopt, flatbuffers::nullopt, 1.0f);
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_SWISH_EXT);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        PointwiseAttrsHolder h(PM::SWISH_FWD, flatbuffers::nullopt, flatbuffers::nullopt, 1.0f);
        auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT);
    }
    {
        PointwiseAttrsHolder h(PM::SWISH_FWD, flatbuffers::nullopt, flatbuffers::nullopt, 2.0f);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }

    // -- Unsupported operations → throw --------------------------------------
    {
        PointwiseAttrsHolder h(PM::ADD);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    {
        PointwiseAttrsHolder h(PM::IDENTITY);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

// ============================================================================
// tensorDataTypeToHipDataType
// ============================================================================

TEST(TestHipblasltUtils, TensorDataTypeToHipblasltDataType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::FLOAT), HIP_R_32F);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::INT32), HIP_R_32I);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::HALF), HIP_R_16F);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::BFLOAT16), HIP_R_16BF);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::INT8), HIP_R_8I);
}

TEST(TestHipblasltUtils, TensorDataTypeToHipblasltDataTypeThrowsOnUnsupported)
{
    // Use a value not in the enum
    EXPECT_THROW(hipblaslt_utils::tensorDataTypeToHipDataType(
                     static_cast<hipdnn_flatbuffers_sdk::data_objects::DataType>(-1)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// FindDeviceBuffer
// ============================================================================

TEST(TestHipblasltUtils, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{42, reinterpret_cast<void*>(0x1234)}, {99, reinterpret_cast<void*>(0x5678)}};

    auto result = hipblaslt_utils::findDeviceBuffer(99, buffers.data(), 2);
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(TestHipblasltUtils, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(
        hipblaslt_utils::findDeviceBuffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// FindTensorAttributes
// ============================================================================

TEST(TestHipblasltUtils, FindTensorAttributesReturnsCorrectValue)
{
    flatbuffers::FlatBufferBuilder builder1;
    auto attrOffset1
        = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(builder1, 1);
    builder1.Finish(attrOffset1);

    flatbuffers::FlatBufferBuilder builder2;
    auto attrOffset2
        = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(builder2, 2);
    builder2.Finish(attrOffset2);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder1.GetBufferPointer());
    auto attrPtr2 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder2.GetBufferPointer());

    auto attrMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{
            {1, attrPtr1}, {2, attrPtr2}};

    EXPECT_EQ(hipblaslt_utils::findTensorAttributes(attrMap, 1).uid(), 1);
    EXPECT_EQ(hipblaslt_utils::findTensorAttributes(attrMap, 2).uid(), 2);
}

TEST(TestHipblasltUtils, FindTensorAttributesThrowsIfNotFound)
{
    auto attrMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{};

    EXPECT_THROW(hipblaslt_utils::findTensorAttributes(attrMap, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
