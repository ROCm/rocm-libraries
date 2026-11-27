// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltUtils.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

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
                         flatbuffers::Optional<float> reluLowerClipSlope = flatbuffers::nullopt,
                         flatbuffers::Optional<float> swishBeta = flatbuffers::nullopt)
    {
        const auto offset
            = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(builder,
                                                                              operation,
                                                                              reluLowerClip,
                                                                              reluUpperClip,
                                                                              reluLowerClipSlope,
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
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(nullptr, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_DEFAULT);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(nullptr, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_BIAS);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }

    // -- ReLU / Clamp --------------------------------------------------------
    {
        // RELU_FWD with both clips → CLAMP without bias
        const PointwiseAttrsHolder h(PM::RELU_FWD, -1.0f, 6.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_CLAMP_EXT);
        EXPECT_FLOAT_EQ(result.act0, -1.0f);
        EXPECT_FLOAT_EQ(result.act1, 6.0f);
    }
    {
        // RELU_FWD with both clips → CLAMP with bias
        const PointwiseAttrsHolder h(PM::RELU_FWD, -1.0f, 6.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT);
        EXPECT_FLOAT_EQ(result.act0, -1.0f);
        EXPECT_FLOAT_EQ(result.act1, 6.0f);
    }
    {
        // RELU_FWD standard (lower_clip = 0, no upper_clip) without bias
        const PointwiseAttrsHolder h(PM::RELU_FWD, 0.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_RELU);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        // RELU_FWD standard (lower_clip = 0, no upper_clip) with bias
        const PointwiseAttrsHolder h(PM::RELU_FWD, 0.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_RELU_BIAS);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        // RELU_FWD with non-zero lower_clip only → throws
        const PointwiseAttrsHolder h(PM::RELU_FWD, 1.0f);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    {
        // RELU_FWD with nothing set: unset lower clip means 0, so this is standard ReLU
        const PointwiseAttrsHolder h(PM::RELU_FWD);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_RELU);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        const PointwiseAttrsHolder h(PM::RELU_FWD);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_RELU_BIAS);
    }
    {
        // A slope of exactly 0 is the same as no slope
        const PointwiseAttrsHolder h(
            PM::RELU_FWD, flatbuffers::nullopt, flatbuffers::nullopt, 0.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_RELU);
    }
    {
        // Leaky ReLU encoded with an explicit zero lower clip must not run as plain ReLU
        const PointwiseAttrsHolder h(PM::RELU_FWD, 0.0f, flatbuffers::nullopt, 0.01f);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    {
        const PointwiseAttrsHolder h(
            PM::RELU_FWD, flatbuffers::nullopt, flatbuffers::nullopt, 0.01f);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    {
        // Upper clip only (ReLU6): unset lower clip means 0
        const PointwiseAttrsHolder h(PM::RELU_FWD, flatbuffers::nullopt, 6.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_CLAMP_EXT);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 6.0f);
    }
    {
        const PointwiseAttrsHolder h(PM::RELU_FWD, flatbuffers::nullopt, 6.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 6.0f);
    }
    {
        const PointwiseAttrsHolder h(PM::RELU_FWD, -1.0f, 6.0f, 0.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_CLAMP_EXT);
        EXPECT_FLOAT_EQ(result.act0, -1.0f);
        EXPECT_FLOAT_EQ(result.act1, 6.0f);
    }
    {
        const PointwiseAttrsHolder h(PM::RELU_FWD, -1.0f, 6.0f, 0.01f);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }

    // -- GELU ----------------------------------------------------------------
    {
        const PointwiseAttrsHolder h(PM::GELU_APPROX_TANH_FWD);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_GELU);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        const PointwiseAttrsHolder h(PM::GELU_APPROX_TANH_FWD);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_GELU_BIAS);
    }

    // -- Swish ---------------------------------------------------------------
    {
        const PointwiseAttrsHolder h(
            PM::SWISH_FWD, flatbuffers::nullopt, flatbuffers::nullopt, flatbuffers::nullopt, 1.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_SWISH_EXT);
        EXPECT_FLOAT_EQ(result.act0, 0.0f);
        EXPECT_FLOAT_EQ(result.act1, 0.0f);
    }
    {
        const PointwiseAttrsHolder h(
            PM::SWISH_FWD, flatbuffers::nullopt, flatbuffers::nullopt, flatbuffers::nullopt, 1.0f);
        const auto result = hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, true);
        EXPECT_EQ(result.epilogue, HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT);
    }
    {
        const PointwiseAttrsHolder h(
            PM::SWISH_FWD, flatbuffers::nullopt, flatbuffers::nullopt, flatbuffers::nullopt, 2.0f);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }

    // -- Unsupported operations → throw --------------------------------------
    {
        const PointwiseAttrsHolder h(PM::ADD);
        EXPECT_THROW(hipblaslt_utils::mapPointwiseModeToHipblasLtEpilogue(h.attrs, false),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    {
        const PointwiseAttrsHolder h(PM::IDENTITY);
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
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::FP8_E4M3), HIP_R_8F_E4M3);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::FP8_E5M2), HIP_R_8F_E5M2);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::FP4_E2M1), HIP_R_4F_E2M1);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::FP6_E2M3), HIP_R_6F_E2M3);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::FP6_E3M2), HIP_R_6F_E3M2);
}

TEST(TestHipblasltUtils, TensorDataTypeToHipblasltDataTypeThrowsOnUnsupported)
{
    // Use a value not in the enum
    EXPECT_THROW(hipblaslt_utils::tensorDataTypeToHipDataType(
                     static_cast<hipdnn_flatbuffers_sdk::data_objects::DataType>(-1)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// operandTranspose / requireRowMajorOutput
// ============================================================================

namespace
{
// Builds a FLOAT tensor with the given dims and strides and keeps the builder alive.
struct TensorHolder
{
    flatbuffers::FlatBufferBuilder builder;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attrs = nullptr;

    TensorHolder(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
    {
        builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            1,
            "t",
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            &strides,
            &dims));
        attrs = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
            builder.GetBufferPointer());
    }
};
} // namespace

TEST(TestHipblasltUtils, OperandTransposeFollowsMatrixOrder)
{
    struct Row
    {
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
        hipblasOperation_t expected;
    };

    const std::vector<Row> rows = {
        {{4, 8}, {8, 1}, HIPBLAS_OP_N},
        {{4, 8}, {1, 4}, HIPBLAS_OP_T},
        // A [M, 1] operand with column-style strides is row-major: its size-1 stride is ignored.
        {{4, 1}, {1, 4}, HIPBLAS_OP_N},
        // A [1, K] operand whose strides are neither 1 is column-major by its size-1 dim.
        {{1, 6}, {99, 100}, HIPBLAS_OP_T},
    };

    for(const auto& row : rows)
    {
        const TensorHolder holder(row.dims, row.strides);
        EXPECT_EQ(
            operandTranspose(hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper(
                holder.attrs)),
            row.expected)
            << "dims " << row.dims[0] << "x" << row.dims[1] << " strides " << row.strides[0] << ","
            << row.strides[1];
    }
}

TEST(TestHipblasltUtils, OperandTransposeThrowsOnUnsupportedLayout)
{
    const TensorHolder holder({4, 8}, {16, 2});
    EXPECT_THROW(
        operandTranspose(
            hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper(holder.attrs)),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipblasltUtils, RequireRowMajorOutputAcceptsOnlyRowMajor)
{
    using hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper;

    const TensorHolder rowMajor({4, 8}, {8, 1});
    EXPECT_NO_THROW(requireRowMajorOutput(TensorAttributesWrapper(rowMajor.attrs)));

    // A [M, 1] output is row-major whatever stride its size-1 dim carries.
    const TensorHolder singleColumn({4, 1}, {1, 4});
    EXPECT_NO_THROW(requireRowMajorOutput(TensorAttributesWrapper(singleColumn.attrs)));

    const TensorHolder columnMajor({4, 8}, {1, 4});
    EXPECT_THROW(requireRowMajorOutput(TensorAttributesWrapper(columnMajor.attrs)),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    const TensorHolder unsupported({4, 8}, {16, 2});
    EXPECT_THROW(requireRowMajorOutput(TensorAttributesWrapper(unsupported.attrs)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipblasltUtils, LeadingDimensionFollowsMatrixOrder)
{
    struct Row
    {
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
        int64_t expected;
    };

    const std::vector<Row> rows = {
        // Row-major: the row stride.
        {{4, 8}, {10, 1}, 10},
        {{2, 4, 8}, {40, 10, 1}, 10},
        // Column-major: the column stride.
        {{4, 8}, {1, 6}, 6},
        {{2, 4, 8}, {48, 1, 6}, 6},
        // A single-row row-major matrix has no row stride: ld is the column count.
        {{1, 8}, {-3, 1}, 8},
        // A size-1 column count is row-major and the stride of that dim is ignored.
        {{4, 1}, {3, -5}, 3},
        // Zero and negative leading dimensions are reported as they are.
        {{4, 8}, {0, 1}, 0},
        {{4, 8}, {1, 0}, 0},
        {{4, 8}, {-2, 1}, -2},
        {{4, 8}, {1, -2}, -2},
    };

    for(const auto& row : rows)
    {
        const TensorHolder holder(row.dims, row.strides);
        EXPECT_EQ(
            leadingDimension(hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper(
                holder.attrs)),
            row.expected)
            << "dims " << row.dims.size() << "d last " << row.dims.back() << " strides "
            << row.strides[row.strides.size() - 2] << "," << row.strides.back();
    }
}

TEST(TestHipblasltUtils, LeadingDimensionThrowsOnUnsupportedLayout)
{
    const TensorHolder holder({4, 8}, {16, 2});
    EXPECT_THROW(
        leadingDimension(
            hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper(holder.attrs)),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

// Fails if an output whose rows overlap (row stride below the column count, zero included) is
// accepted, if a packed or padded output is declined, or if a single-row output, whose row
// stride is never used, is declined for its stride.
TEST(TestHipblasltUtils, RequireRowMajorOutputRejectsOverlappingRows)
{
    using hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper;

    const TensorHolder zeroLd({4, 5}, {0, 1});
    EXPECT_THROW(requireRowMajorOutput(TensorAttributesWrapper(zeroLd.attrs)),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    const TensorHolder shortLd({4, 5}, {4, 1});
    EXPECT_THROW(requireRowMajorOutput(TensorAttributesWrapper(shortLd.attrs)),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    const TensorHolder unitLd({4, 5}, {1, 1});
    EXPECT_THROW(requireRowMajorOutput(TensorAttributesWrapper(unitLd.attrs)),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    const TensorHolder zeroLdBatched({2, 4, 8}, {32, 0, 1});
    EXPECT_THROW(requireRowMajorOutput(TensorAttributesWrapper(zeroLdBatched.attrs)),
                 hipdnn_plugin_sdk::HipdnnPluginException);

    const TensorHolder packed({4, 5}, {5, 1});
    EXPECT_NO_THROW(requireRowMajorOutput(TensorAttributesWrapper(packed.attrs)));

    const TensorHolder padded({4, 5}, {7, 1});
    EXPECT_NO_THROW(requireRowMajorOutput(TensorAttributesWrapper(padded.attrs)));

    const TensorHolder singleRow({1, 5}, {0, 1});
    EXPECT_NO_THROW(requireRowMajorOutput(TensorAttributesWrapper(singleRow.attrs)));
}

// ============================================================================
// FindTensorAttributes
// ============================================================================

TEST(TestHipblasltUtils, FindTensorAttributesReturnsCorrectValue)
{
    flatbuffers::FlatBufferBuilder builder1;
    const auto attrOffset1
        = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(builder1, 1);
    builder1.Finish(attrOffset1);

    flatbuffers::FlatBufferBuilder builder2;
    const auto attrOffset2
        = hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(builder2, 2);
    builder2.Finish(attrOffset2);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder1.GetBufferPointer());
    auto attrPtr2 = flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>(
        builder2.GetBufferPointer());

    const auto attrMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{
            {1, attrPtr1}, {2, attrPtr2}};

    EXPECT_EQ(hipblaslt_utils::findTensorAttributes(attrMap, 1).uid(), 1);
    EXPECT_EQ(hipblaslt_utils::findTensorAttributes(attrMap, 2).uid(), 2);
}

TEST(TestHipblasltUtils, FindTensorAttributesThrowsIfNotFound)
{
    const auto attrMap
        = std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>{};

    EXPECT_THROW(hipblaslt_utils::findTensorAttributes(attrMap, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// IsTypeFp8Ocp
// ============================================================================

TEST(TestHipblasltUtils, IsTypeFp8Ocp)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    EXPECT_TRUE(hipblaslt_utils::isTypeFp8Ocp(DataType::FP8_E4M3));
    EXPECT_TRUE(hipblaslt_utils::isTypeFp8Ocp(DataType::FP8_E5M2));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp8Ocp(DataType::FP8_E8M0));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp8Ocp(DataType::FLOAT));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp8Ocp(DataType::HALF));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp8Ocp(DataType::BFLOAT16));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp8Ocp(DataType::FP4_E2M1));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp8Ocp(DataType::FP6_E2M3));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp8Ocp(DataType::FP6_E3M2));
}

// ============================================================================
// IsTypeFp6Ocp
// ============================================================================

TEST(TestHipblasltUtils, IsTypeFp6Ocp)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    EXPECT_TRUE(hipblaslt_utils::isTypeFp6Ocp(DataType::FP6_E2M3));
    EXPECT_TRUE(hipblaslt_utils::isTypeFp6Ocp(DataType::FP6_E3M2));

    EXPECT_FALSE(hipblaslt_utils::isTypeFp6Ocp(DataType::FP8_E4M3));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp6Ocp(DataType::FP8_E5M2));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp6Ocp(DataType::FP8_E8M0));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp6Ocp(DataType::FP4_E2M1));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp6Ocp(DataType::FLOAT));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp6Ocp(DataType::HALF));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp6Ocp(DataType::BFLOAT16));
    EXPECT_FALSE(hipblaslt_utils::isTypeFp6Ocp(DataType::INT8));
}

// ============================================================================
// IsTypeMxOcp
// ============================================================================

TEST(TestHipblasltUtils, IsTypeMxOcp)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    EXPECT_TRUE(hipblaslt_utils::isTypeMxOcp(DataType::FP8_E4M3));
    EXPECT_TRUE(hipblaslt_utils::isTypeMxOcp(DataType::FP8_E5M2));
    EXPECT_TRUE(hipblaslt_utils::isTypeMxOcp(DataType::FP4_E2M1));
    EXPECT_TRUE(hipblaslt_utils::isTypeMxOcp(DataType::FP6_E2M3));
    EXPECT_TRUE(hipblaslt_utils::isTypeMxOcp(DataType::FP6_E3M2));

    EXPECT_FALSE(hipblaslt_utils::isTypeMxOcp(DataType::FP8_E8M0));
    EXPECT_FALSE(hipblaslt_utils::isTypeMxOcp(DataType::FLOAT));
    EXPECT_FALSE(hipblaslt_utils::isTypeMxOcp(DataType::HALF));
    EXPECT_FALSE(hipblaslt_utils::isTypeMxOcp(DataType::BFLOAT16));
    EXPECT_FALSE(hipblaslt_utils::isTypeMxOcp(DataType::INT8));
}

// ============================================================================
// MatrixOrder
// ============================================================================

TEST(TestHipblasltUtils, MatrixOrderClassifiesLastTwoDims)
{
    struct Row
    {
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
        MatrixOrder expected;
    };

    const std::vector<Row> rows = {
        {{4, 8}, {8, 1}, MatrixOrder::ROW_MAJOR},
        {{4, 8}, {1, 4}, MatrixOrder::COLUMN_MAJOR},
        {{2, 4, 8}, {32, 8, 1}, MatrixOrder::ROW_MAJOR},
        {{2, 4, 8}, {32, 1, 4}, MatrixOrder::COLUMN_MAJOR},
        // A size-1 last dim is row-major whatever its stride says.
        {{8, 1}, {1, 8}, MatrixOrder::ROW_MAJOR},
        {{33, 1}, {1, 33}, MatrixOrder::ROW_MAJOR},
        {{1, 1}, {5, 7}, MatrixOrder::ROW_MAJOR},
        // A size-1 second-to-last dim is column-major only when the last dim is not 1 and
        // strides[rank-1] != 1.
        {{1, 6}, {6, 1}, MatrixOrder::ROW_MAJOR},
        {{1, 6}, {99, 100}, MatrixOrder::COLUMN_MAJOR},
        // Neither stride is 1 and no dim is 1.
        {{4, 8}, {16, 2}, MatrixOrder::UNSUPPORTED},
        // Rank below 2 and dims/strides size mismatch.
        {{8}, {1}, MatrixOrder::UNSUPPORTED},
        {{4, 8}, {1}, MatrixOrder::UNSUPPORTED},
    };

    for(const auto& row : rows)
    {
        EXPECT_EQ(matrixOrder(row.dims, row.strides), row.expected)
            << "rank " << row.dims.size() << " last dim " << row.dims.back();
    }
}

// ============================================================================
// IsDenseRowMajor
// ============================================================================

TEST(TestHipblasltUtils, IsDenseRowMajorAcceptsDenseLayoutsOfAnyRank)
{
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({}, {}));
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({7}, {1}));
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({4, 5}, {5, 1}));
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({2, 3, 4}, {12, 4, 1}));
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({2, 3, 4, 5}, {60, 20, 5, 1}));
}

TEST(TestHipblasltUtils, IsDenseRowMajorDeclinesOtherStrideOrders)
{
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({4, 5}, {1, 4}));
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({2, 3, 4}, {1, 2, 6}));
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({2, 3, 4}, {12, 1, 3}));
    // Padded outer stride, padded inner stride.
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({2, 3, 4}, {16, 4, 1}));
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({2, 3, 4}, {12, 4, 2}));
}

TEST(TestHipblasltUtils, IsDenseRowMajorIgnoresStridesOfSizeOneDimensions)
{
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({1, 5}, {0, 1}));
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({1, 5}, {99, 1}));
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({2, 1, 4}, {4, 123, 1}));
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({1, 1, 1}, {7, 8, 9}));
    // A trailing size-1 dimension does not change the expected stride of the one before it.
    EXPECT_TRUE(hipblaslt_utils::isDenseRowMajor({4, 5, 1}, {5, 1, 0}));
    // A size-1 stride is ignored, but the non-unit dimensions must still be dense.
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({1, 5}, {5, 2}));
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({2, 1, 4}, {5, 1, 1}));
}

TEST(TestHipblasltUtils, IsDenseRowMajorDeclinesRankMismatch)
{
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({4, 5}, {5}));
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({4}, {5, 1}));
    EXPECT_FALSE(hipblaslt_utils::isDenseRowMajor({}, {1}));
}

// ============================================================================
// THROW_ON_HIPBLASLT_FAILURE / LOG_ON_HIPBLASLT_FAILURE
// ============================================================================

TEST(TestHipblasltUtils, ThrowOnHipblasltFailureEvaluatesItsArgumentOnce)
{
    int calls = 0;
    auto failsOnce = [&calls]() {
        return ++calls == 1 ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS;
    };

    try
    {
        THROW_ON_HIPBLASLT_FAILURE(failsOnce());
        FAIL() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_NE(e.getMessage().find("HIPBLAS_STATUS_INVALID_VALUE"), std::string::npos)
            << e.getMessage();
    }
    EXPECT_EQ(calls, 1);
}

TEST(TestHipblasltUtils, ThrowOnHipblasltFailureDoesNotThrowOnSuccess)
{
    int calls = 0;
    auto succeeds = [&calls]() {
        ++calls;
        return HIPBLAS_STATUS_SUCCESS;
    };

    EXPECT_NO_THROW(THROW_ON_HIPBLASLT_FAILURE(succeeds()));
    EXPECT_EQ(calls, 1);
}

TEST(TestHipblasltUtils, LogOnHipblasltFailureEvaluatesItsArgumentOnce)
{
    hipdnn_plugin_sdk::logging::initializeCallbackLogging(
        "hipblaslt_plugin_tests", hipdnn_test_sdk::utilities::initializeTestLogRecordingShared());
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    int calls = 0;
    auto failsOnce = [&calls]() {
        return ++calls == 1 ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS;
    };

    LOG_ON_HIPBLASLT_FAILURE(failsOnce());

    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "HIPBLAS_STATUS_INVALID_VALUE"))
        << recorder.getRecordedLogsAsString();
}
