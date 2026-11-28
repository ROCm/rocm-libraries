// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestTensorAttributes, DefaultConstructor)
{
    const TensorAttributes tensor;
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_EQ(tensor.get_name(), "");
    EXPECT_EQ(tensor.get_data_type(), DataType::NOT_SET);
    EXPECT_TRUE(tensor.get_stride().empty());
    EXPECT_TRUE(tensor.get_dim().empty());
    EXPECT_EQ(tensor.get_volume(), 1);
    EXPECT_FALSE(tensor.get_is_virtual());
    EXPECT_FALSE(tensor.has_uid());
}

TEST(TestTensorAttributes, SetAndGetUid)
{
    TensorAttributes tensor;
    tensor.set_uid(42);
    EXPECT_EQ(tensor.get_uid(), 42);
    EXPECT_TRUE(tensor.has_uid());

    tensor.clear_uid();
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_FALSE(tensor.has_uid());
}

// cuDNN-parity nested alias: consumer source spells variant-pack map keys as
// TensorAttributes::uid_t, and it must be the same type get_uid() returns.
TEST(TestTensorAttributes, UidTypeAliasMatchesUidAccessors)
{
    static_assert(std::is_same_v<TensorAttributes::uid_t, int64_t>);
    static_assert(std::is_same_v<TensorAttributes::uid_t,
                                 decltype(std::declval<TensorAttributes>().get_uid())>);

    std::unordered_map<TensorAttributes::uid_t, void*> variantPack;
    TensorAttributes tensor;
    tensor.set_uid(7);
    variantPack.emplace(tensor.get_uid(), nullptr);

    EXPECT_EQ(variantPack.count(7), 1U);
}

TEST(TestTensorAttributes, SetAndGetName)
{
    TensorAttributes tensor;
    tensor.set_name("TestTensor");
    EXPECT_EQ(tensor.get_name(), "TestTensor");
}

TEST(TestTensorAttributes, SetAndGetDataType)
{
    TensorAttributes tensor;
    tensor.set_data_type(DataType::FLOAT);
    EXPECT_EQ(tensor.get_data_type(), DataType::FLOAT);
}

TEST(TestTensorAttributes, SetAndGetStride)
{
    TensorAttributes tensor;
    tensor.set_stride({1, 2, 3});
    EXPECT_EQ(tensor.get_stride(), std::vector<int64_t>({1, 2, 3}));
}

TEST(TestTensorAttributes, SetAndGetDim)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    EXPECT_EQ(tensor.get_dim(), std::vector<int64_t>({4, 5, 6}));
    EXPECT_EQ(tensor.get_volume(), 4 * 5 * 6);
}

TEST(TestTensorAttributes, SetAndGetIsVirtual)
{
    TensorAttributes tensor;
    tensor.set_is_virtual(true);
    EXPECT_TRUE(tensor.get_is_virtual());

    tensor.set_is_virtual(false);
    EXPECT_FALSE(tensor.get_is_virtual());
}

TEST(TestTensorAttributes, SetOutput)
{
    TensorAttributes tensor;
    tensor.set_output(true);
    EXPECT_FALSE(tensor.get_is_virtual());

    tensor.set_output(false);
    EXPECT_TRUE(tensor.get_is_virtual());
}

TEST(TestTensorAttributes, SetFromGraphAttributes)
{
    GraphAttributes graphAttributes;
    graphAttributes.set_io_data_type(DataType::FLOAT);
    graphAttributes.set_intermediate_data_type(DataType::HALF);

    TensorAttributes tensor;
    tensor.set_is_virtual(false).fill_from_context(graphAttributes);
    EXPECT_EQ(tensor.get_data_type(), DataType::FLOAT);

    tensor.set_data_type(DataType::NOT_SET);
    tensor.set_is_virtual(true).fill_from_context(graphAttributes);
    EXPECT_EQ(tensor.get_data_type(), DataType::HALF);
}

TEST(TestTensorAttributes, ValidateSucceedsOnValueTensor)
{
    const TensorAttributes tensor(1.f);
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateSucceedsOnVirtualTensor)
{
    TensorAttributes tensor;
    tensor.set_dim({1});
    tensor.set_stride({1});
    tensor.set_is_virtual(true);
    tensor.set_data_type(DataType::FLOAT);

    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateFailsOnVirtualValueTensor)
{
    TensorAttributes tensor(1.f);
    tensor.set_dim({1});
    tensor.set_stride({1});
    tensor.set_is_virtual(true);

    EXPECT_EQ(tensor.validate(),
              Error(ErrorCode::INVALID_VALUE, "Tensor  cannot be virtual and pass by value"));
}

TEST(TestTensorAttributes, ValidateFailsOnDifferentDimAndStrideSize)
{
    TensorAttributes tensor;
    tensor.set_dim({1});
    tensor.set_stride({1, 2});
    tensor.set_data_type(DataType::FLOAT);

    EXPECT_EQ(tensor.validate(),
              Error(ErrorCode::INVALID_VALUE, "Tensor  dims and strides have different sizes"));
}

TEST(TestTensorAttributes, ValidateFailsOnEmptyDims)
{
    TensorAttributes tensor;
    tensor.set_data_type(DataType::FLOAT);

    EXPECT_EQ(tensor.validate(),
              Error(ErrorCode::ATTRIBUTE_NOT_SET, "Tensor  dims must be non-empty"));
}

TEST(TestTensorAttributes, ValidateFailsOnNonPositiveDimension)
{
    const std::vector<std::vector<int64_t>> testDims
        = {{0, 1}, {1, 0, 1}, {-1, 1, 1}, {1, 1, 1, -1}};

    for(const auto& dim : testDims)
    {
        TensorAttributes tensor;
        tensor.set_dim(dim);
        tensor.set_stride(dim);
        tensor.set_data_type(DataType::FLOAT);

        EXPECT_EQ(tensor.validate(),
                  Error(ErrorCode::INVALID_VALUE, "Tensor  must have only positive dimensions"))
            << "Dims: " << hipdnn_data_sdk::utilities::vecToString(dim);
    }
}

TEST(TestTensorAttributes, ValidateFailsOnNegativeStride)
{
    const std::vector<std::pair<std::vector<int64_t>, std::string>> cases
        = {{{-1, 1, 1}, "Tensor x strides must be non-negative, got -1 on axis 0"},
           {{6, -2, 1}, "Tensor x strides must be non-negative, got -2 on axis 1"},
           {{6, 3, -1}, "Tensor x strides must be non-negative, got -1 on axis 2"}};

    for(const auto& [stride, expectedMessage] : cases)
    {
        TensorAttributes tensor;
        tensor.set_name("x");
        tensor.set_dim({2, 2, 3});
        tensor.set_stride(stride);
        tensor.set_data_type(DataType::FLOAT);

        const auto err = tensor.validate();
        EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE)
            << "Strides: " << hipdnn_data_sdk::utilities::vecToString(stride);
        EXPECT_EQ(err.get_message(), expectedMessage)
            << "Strides: " << hipdnn_data_sdk::utilities::vecToString(stride);
    }
}

TEST(TestTensorAttributes, ValidateReportsLowestNegativeStrideAxis)
{
    TensorAttributes tensor;
    tensor.set_name("x");
    tensor.set_dim({2, 2, 3});
    tensor.set_stride({-6, -3, 1});
    tensor.set_data_type(DataType::FLOAT);

    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_EQ(err.get_message(), "Tensor x strides must be non-negative, got -6 on axis 0");
}

TEST(TestTensorAttributes, ValidateSucceedsWithZeroAndPositiveStrides)
{
    const std::vector<std::vector<int64_t>> testStrides
        = {{0, 0, 1}, {0, 0, 0}, {6, 3, 1}, {1, 6, 2}};

    for(const auto& stride : testStrides)
    {
        TensorAttributes tensor;
        tensor.set_name("x");
        tensor.set_dim({2, 2, 3});
        tensor.set_stride(stride);
        tensor.set_data_type(DataType::FLOAT);

        const auto err = tensor.validate();
        EXPECT_EQ(err.code, ErrorCode::OK)
            << "Strides: " << hipdnn_data_sdk::utilities::vecToString(stride) << ", "
            << err.get_message();
    }
}

TEST(TestTensorAttributes, ValidateReportsNonPositiveDimensionBeforeNegativeStride)
{
    TensorAttributes tensor;
    tensor.set_name("x");
    tensor.set_dim({2, 0, 3});
    tensor.set_stride({6, -3, 1});
    tensor.set_data_type(DataType::FLOAT);

    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_EQ(err.get_message(), "Tensor x must have only positive dimensions");
}

TEST(TestTensorAttributes, ValidateDataType)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({0, 1, 2});

    const std::vector<std::pair<DataType, ErrorCode>> expectedResults
        = {{DataType::NOT_SET, ErrorCode::ATTRIBUTE_NOT_SET},
           {DataType::FLOAT, ErrorCode::OK},
           {DataType::HALF, ErrorCode::OK},
           {DataType::BFLOAT16, ErrorCode::OK},
           {DataType::DOUBLE, ErrorCode::OK},
           {DataType::UINT8, ErrorCode::OK},
           {DataType::INT32, ErrorCode::OK},
           {DataType::INT8, ErrorCode::OK},
           {DataType::FP8_E4M3, ErrorCode::OK},
           {DataType::FP8_E5M2, ErrorCode::OK},
           {DataType::FP8_E8M0, ErrorCode::OK},
           {DataType::FP4_E2M1, ErrorCode::OK},
           {DataType::INT4, ErrorCode::OK},
           {DataType::FP6_E2M3, ErrorCode::OK},
           {DataType::FP6_E3M2, ErrorCode::OK},
           {DataType::INT64, ErrorCode::OK},
           {DataType::BOOLEAN, ErrorCode::OK},
           {DataType::FP8_E4M3_FNUZ, ErrorCode::OK},
           {DataType::FP8_E5M2_FNUZ, ErrorCode::OK},
           {DataType::INT8x4, ErrorCode::INVALID_VALUE},
           {DataType::UINT8x4, ErrorCode::INVALID_VALUE},
           {DataType::INT8x32, ErrorCode::INVALID_VALUE},
           {DataType::FAST_FLOAT_FOR_FP8, ErrorCode::INVALID_VALUE},
           {DataType::COMPLEX_FP32, ErrorCode::INVALID_VALUE},
           {DataType::COMPLEX_FP64, ErrorCode::INVALID_VALUE}};

    for(auto [dataType, errorCode] : expectedResults)
    {
        tensor.set_data_type(dataType);
        auto result = tensor.validate();
        EXPECT_EQ(result.code, errorCode) << "For " + std::string(to_string(dataType));
    }
}

// validate() and the lowering share one table (toHipdnnDataType): a data type
// validate() accepts is one the lowering can express, and vice versa. The probed values run
// past the last enumerator, so an enumerator added later is covered too.
TEST(TestTensorAttributes, ValidateDataTypeAgreesWithLoweringForEveryValue)
{
    constexpr int K_DATA_TYPE_PROBE_END = 64;

    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, 6, 1});

    for(int i = 0; i < K_DATA_TYPE_PROBE_END; ++i)
    {
        const auto dataType = static_cast<DataType>(i);
        tensor.set_data_type(dataType);
        const auto result = tensor.validate();

        ErrorCode expected = ErrorCode::INVALID_VALUE;
        if(dataType == DataType::NOT_SET)
        {
            expected = ErrorCode::ATTRIBUTE_NOT_SET;
        }
        else if(toHipdnnDataType(dataType).has_value())
        {
            expected = ErrorCode::OK;
        }
        EXPECT_EQ(result.code, expected) << "For DataType value " << i;
    }

    // The last probed value is past every enumerator: it has no name and validate() rejects it.
    // Raise K_DATA_TYPE_PROBE_END when this fails.
    const auto pastTheEnd = static_cast<DataType>(K_DATA_TYPE_PROBE_END - 1);
    EXPECT_STREQ(to_string(pastTheEnd), "unknown");
    tensor.set_data_type(pastTheEnd);
    const auto result = tensor.validate();
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("data type unknown is not supported by hipDNN lowering"),
              std::string::npos)
        << result.err_msg;
}

TEST(TestTensorAttributes, ValidateRejectsUnmappedDataTypeWithMessage)
{
    TensorAttributes tensor;
    tensor.set_name("x");
    tensor.set_dim({4});
    tensor.set_stride({1});
    tensor.set_data_type(DataType::INT8x4);

    EXPECT_EQ(tensor.validate(),
              Error(ErrorCode::INVALID_VALUE,
                    "Tensor x data type int8x4 is not supported by hipDNN lowering"));
}

TEST(TestTensorAttributes, SetAndGetRaggedOffset)
{
    TensorAttributes tensor;
    EXPECT_FALSE(tensor.has_ragged_offset());
    EXPECT_EQ(tensor.get_ragged_offset(), nullptr);

    auto aux = std::make_shared<TensorAttributes>();
    aux->set_uid(99);
    tensor.set_ragged_offset(aux);
    EXPECT_TRUE(tensor.has_ragged_offset());
    EXPECT_EQ(tensor.get_ragged_offset()->get_uid(), 99);

    tensor.set_ragged_offset(nullptr);
    EXPECT_FALSE(tensor.has_ragged_offset());
    EXPECT_EQ(tensor.get_ragged_offset(), nullptr);
}

TEST(TestTensorAttributes, SetAndGetAlignment)
{
    TensorAttributes tensor;
    EXPECT_EQ(tensor.get_alignment(), 16);

    tensor.set_alignment(64);
    EXPECT_EQ(tensor.get_alignment(), 64);

    tensor.set_alignment(1);
    EXPECT_EQ(tensor.get_alignment(), 1);
}

TEST(TestTensorAttributes, ValidateFailsOnAlignmentZero)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    tensor.set_alignment(0);
    EXPECT_EQ(tensor.validate().code, ErrorCode::INVALID_VALUE);
}

TEST(TestTensorAttributes, ValidateFailsOnAlignmentNegative)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    tensor.set_alignment(-1);
    EXPECT_EQ(tensor.validate().code, ErrorCode::INVALID_VALUE);
}

TEST(TestTensorAttributes, ValidateSucceedsWithDefaultAlignment)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, RaggedOffsetMethodChainingReturnsThis)
{
    TensorAttributes tensor;
    auto aux = std::make_shared<TensorAttributes>();
    const TensorAttributes& ref = tensor.set_ragged_offset(aux);
    EXPECT_EQ(&ref, &tensor);

    const TensorAttributes& ref2 = tensor.set_alignment(32);
    EXPECT_EQ(&ref2, &tensor);
}

TEST(TestTensorAttributes, SetAndGetRaggedOffsetMultiplier)
{
    TensorAttributes tensor;
    EXPECT_EQ(tensor.get_ragged_offset_multiplier(), 1);
    EXPECT_FALSE(tensor.has_ragged_offset_multiplier());

    tensor.set_ragged_offset_multiplier(512);
    EXPECT_EQ(tensor.get_ragged_offset_multiplier(), 512);
    EXPECT_TRUE(tensor.has_ragged_offset_multiplier());

    tensor.set_ragged_offset_multiplier(1);
    EXPECT_FALSE(tensor.has_ragged_offset_multiplier());

    const TensorAttributes& ref = tensor.set_ragged_offset_multiplier(64);
    EXPECT_EQ(&ref, &tensor);
}

TEST(TestTensorAttributes, ValidateFailsOnRaggedOffsetMultiplierBelowOne)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    auto aux = std::make_shared<TensorAttributes>();
    tensor.set_ragged_offset(aux);
    tensor.set_ragged_offset_multiplier(0);
    EXPECT_EQ(tensor.validate().code, ErrorCode::INVALID_VALUE);
}

TEST(TestTensorAttributes, ValidateFailsOnMultiplierWithoutRaggedOffset)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    tensor.set_ragged_offset_multiplier(512);
    EXPECT_EQ(tensor.validate().code, ErrorCode::INVALID_VALUE);
}

TEST(TestTensorAttributes, ValidateSucceedsWithMultiplierAndRaggedOffset)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    auto aux = std::make_shared<TensorAttributes>();
    tensor.set_ragged_offset(aux);
    tensor.set_ragged_offset_multiplier(512);
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateFailsOnNegativeRaggedOffsetStride)
{
    TensorAttributes tensor;
    tensor.set_name("x");
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    auto offset = std::make_shared<TensorAttributes>();
    offset->set_name("offs");
    offset->set_stride({-1, 1, 1, 1});
    tensor.set_ragged_offset(offset);

    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_EQ(err.get_message(),
              "Tensor x ragged offset offs strides must be non-negative, got -1 on axis 0");
}

TEST(TestTensorAttributes, ValidateSucceedsWithZeroStrideRaggedOffset)
{
    TensorAttributes tensor;
    tensor.set_name("x");
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    auto offset = std::make_shared<TensorAttributes>();
    offset->set_name("offs");
    offset->set_stride({0, 1, 1, 1});
    tensor.set_ragged_offset(offset);

    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::OK) << err.get_message();
}

TEST(TestTensorAttributes, ValidateSucceedsWithDefaultRaggedOffsetMultiplier)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 1, 1, 1});
    tensor.set_stride({1, 1, 1, 1});
    tensor.set_data_type(DataType::FLOAT);
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}
TEST(TestTensorAttributes, ValidateSucceedsOnRuntimeWithDefaultTensor)
{
    // flag true + value present; set_value seeded dims/strides/data_type.
    TensorAttributes tensor(1.F);
    tensor.set_is_pass_by_value(true);
    ASSERT_TRUE(tensor.get_is_runtime_pass_by_value());
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateSucceedsOnRuntimeUserSuppliedTensor)
{
    // flag true + value cleared; dims/strides/data_type survive set_value seeding.
    TensorAttributes tensor(1.F);
    tensor.set_as_runtime_parameter();
    ASSERT_TRUE(tensor.get_is_runtime_pass_by_value());
    EXPECT_TRUE(std::holds_alternative<std::monostate>(tensor.get_value_variant()));
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateFailsOnVirtualRuntimePassByValueTensor)
{
    // virtual + runtime flag (no stored value) is the flag-exclusion case.
    TensorAttributes tensor;
    tensor.set_dim({1});
    tensor.set_stride({1});
    tensor.set_data_type(DataType::FLOAT);
    tensor.set_as_runtime_parameter();
    tensor.set_is_virtual(true);

    EXPECT_EQ(
        tensor.validate(),
        Error(ErrorCode::INVALID_VALUE, "Tensor  cannot be virtual and runtime pass by value"));
}

namespace
{
TensorAttributes makeRaggedTensor(const std::vector<int64_t>& dims,
                                  const std::vector<int64_t>& strides)
{
    TensorAttributes tensor;
    tensor.set_name("Q").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    auto offset = std::make_shared<TensorAttributes>();
    offset->set_dim({dims[0] + 1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(DataType::INT32);
    tensor.set_ragged_offset(offset);
    return tensor;
}
} // namespace

TEST(TestTensorAttributes, ValidateSucceedsOnPackedRaggedTensor)
{
    // BSHD packing of B = 2, H = 4, S = 64, D = 128.
    const auto tensor = makeRaggedTensor({2, 4, 64, 128}, {32768, 128, 512, 1});
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateIgnoresRaggedBatchStride)
{
    const auto tensor = makeRaggedTensor({2, 4, 64, 128}, {1, 128, 512, 1});
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateFailsOnRaggedZeroSequenceStride)
{
    const auto tensor = makeRaggedTensor({2, 4, 64, 128}, {32768, 128, 0, 1});
    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_THAT(
        err.get_message(),
        ::testing::HasSubstr("Ragged tensor Q sequence stride (axis 2) must be positive, got 0"));
}

TEST(TestTensorAttributes, ValidateFailsOnRaggedNegativeSequenceStride)
{
    const auto tensor = makeRaggedTensor({2, 4, 64, 128}, {32768, 128, -512, 1});
    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_THAT(err.get_message(), ::testing::HasSubstr("must be positive, got -512"));
}

TEST(TestTensorAttributes, ValidateFailsOnRaggedOverlappingTokenRows)
{
    // Token row t covers elements 64t to 64t + 127, so neighbouring rows share 64 elements.
    const auto tensor = makeRaggedTensor({2, 1, 64, 128}, {8192, 8192, 64, 1});
    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_THAT(err.get_message(),
                ::testing::HasSubstr("Ragged tensor Q violates the sorted-stride rule: axis 2 "
                                     "has stride 64, below the span 128"));
}

TEST(TestTensorAttributes, ValidateFailsOnRaggedOverlappingHeads)
{
    const auto tensor = makeRaggedTensor({2, 4, 64, 128}, {32768, 64, 512, 1});
    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_THAT(err.get_message(),
                ::testing::HasSubstr("axis 1 has stride 64, below the span 128"));
}

TEST(TestTensorAttributes, ValidateFailsOnRaggedHeadMajorStrideBelowSequenceSpan)
{
    // [H, T] layout with a head stride of 5 tokens and a declared sequence extent of 8.
    const auto tensor = makeRaggedTensor({2, 4, 8, 1}, {5, 5, 1, 1});
    const auto err = tensor.validate();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
    EXPECT_THAT(err.get_message(), ::testing::HasSubstr("axis 1 has stride 5, below the span 8"));
}

TEST(TestTensorAttributes, ValidateSucceedsOnRaggedHeadMajorLayout)
{
    // [H, T] layout: B = 2, H = 4, S = 3, five tokens in total.
    const auto tensor = makeRaggedTensor({2, 4, 3, 1}, {20, 5, 1, 1});
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateSucceedsOnRaggedZeroHeadStride)
{
    // Every head reads the same storage.
    const auto tensor = makeRaggedTensor({2, 4, 64, 128}, {8192, 0, 128, 1});
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateSucceedsOnRaggedOverlapOnlyThroughSizeOneAxes)
{
    EXPECT_EQ(makeRaggedTensor({2, 1, 64, 1}, {64, 1, 1, 1}).validate(), Error(ErrorCode::OK, ""));
    EXPECT_EQ(makeRaggedTensor({2, 1, 64, 8}, {512, 4, 8, 1}).validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateChecksRaggedLayoutFromRankThree)
{
    const auto rank3 = makeRaggedTensor({2, 4, 64}, {256, 64, 0});
    EXPECT_EQ(rank3.validate().code, ErrorCode::INVALID_VALUE);

    const auto rank2 = makeRaggedTensor({2, 64}, {0, 0});
    EXPECT_EQ(rank2.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateIgnoresRaggedLayoutRulesOnNonRaggedTensor)
{
    for(const auto& strides : {std::vector<int64_t>{32768, 128, 0, 1},
                               std::vector<int64_t>{32768, 64, 512, 1},
                               std::vector<int64_t>{8192, 8192, 64, 1}})
    {
        TensorAttributes tensor;
        tensor.set_dim({2, 4, 64, 128}).set_stride(strides).set_data_type(DataType::FLOAT);
        EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
    }
}

TEST(TestTensorAttributes, TensorLogicalAndStrictEquality)
{
    TensorAttributes tensorA;
    tensorA.set_dim({1, 64, 28, 28});
    tensorA.set_stride({50176, 784, 28, 1});
    tensorA.set_data_type(DataType::HALF);
    tensorA.set_uid(1);
    tensorA.set_name("Tensor_A");

    TensorAttributes tensorB;
    tensorB.set_dim({1, 64, 28, 28});
    tensorB.set_stride({50176, 784, 28, 1});
    tensorB.set_data_type(DataType::HALF);
    tensorB.set_uid(2);
    tensorB.set_name("Tensor_B");

    EXPECT_TRUE(tensorA.logicallyEquals(tensorB));
    EXPECT_TRUE(tensorB.logicallyEquals(tensorA));

    EXPECT_FALSE(tensorA == tensorB);
    EXPECT_TRUE(tensorA != tensorB);

    tensorB.set_name("Tensor_A");
    tensorB.set_uid(1);
    EXPECT_TRUE(tensorA == tensorB);
    EXPECT_FALSE(tensorA != tensorB);

    tensorB.set_is_virtual(true);
    EXPECT_FALSE(tensorA.logicallyEquals(tensorB));
    EXPECT_FALSE(tensorA == tensorB);

    const TensorAttributes scalarA(2.5f);
    const TensorAttributes scalarB(2.5f);
    const TensorAttributes scalarC(3.5f);

    EXPECT_TRUE(scalarA.logicallyEquals(scalarB));
    EXPECT_FALSE(scalarA.logicallyEquals(scalarC));

    // Same value, different pass-by-value mode: runtime-with-default (floors the
    // provider at 1.2.0) vs. compile-time constant (1.0.0) are not interchangeable
    // even though the baked value matches.
    const TensorAttributes runtimeWithDefault(2.5f, ScalarType::RUNTIME_PARAM);
    const TensorAttributes compileTimeConstant(2.5f, ScalarType::COMPILE_TIME_CONST);
    EXPECT_TRUE(runtimeWithDefault.get_is_runtime_pass_by_value());
    EXPECT_FALSE(compileTimeConstant.get_is_runtime_pass_by_value());
    EXPECT_FALSE(runtimeWithDefault.logicallyEquals(compileTimeConstant));
    EXPECT_FALSE(compileTimeConstant.logicallyEquals(runtimeWithDefault));
}

namespace
{

TensorAttributes makeEqualityTensor(int64_t uid, const std::string& name)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 16, 8, 64});
    tensor.set_stride({8192, 512, 64, 1});
    tensor.set_data_type(DataType::HALF);
    tensor.set_uid(uid);
    tensor.set_name(name);
    return tensor;
}

std::shared_ptr<TensorAttributes>
    makeRaggedOffset(int64_t batch, int64_t uid, const std::string& name)
{
    auto offset = std::make_shared<TensorAttributes>();
    offset->set_dim({batch, 1, 1, 1});
    offset->set_stride({1, 1, 1, 1});
    offset->set_data_type(DataType::INT32);
    offset->set_uid(uid);
    offset->set_name(name);
    return offset;
}

} // namespace

TEST(TestTensorAttributes, LogicalEqualityDistinguishesAlignment)
{
    auto tensorA = makeEqualityTensor(1, "t");
    auto tensorB = makeEqualityTensor(1, "t");
    ASSERT_TRUE(tensorA == tensorB);

    tensorB.set_alignment(4);
    EXPECT_FALSE(tensorA.logicallyEquals(tensorB));
    EXPECT_FALSE(tensorB.logicallyEquals(tensorA));
    EXPECT_FALSE(tensorA == tensorB);
}

TEST(TestTensorAttributes, LogicalEqualityDistinguishesRaggedOffset)
{
    auto ragged = makeEqualityTensor(1, "t");
    ragged.set_ragged_offset(makeRaggedOffset(4, 10, "offs"));
    const auto plain = makeEqualityTensor(1, "t");

    // Ragged vs non-ragged, both directions.
    EXPECT_FALSE(ragged.logicallyEquals(plain));
    EXPECT_FALSE(plain.logicallyEquals(ragged));
    EXPECT_FALSE(ragged == plain);
    EXPECT_FALSE(plain == ragged);

    // Offset tensors with different layouts.
    auto otherLayout = makeEqualityTensor(1, "t");
    otherLayout.set_ragged_offset(makeRaggedOffset(3, 10, "offs"));
    EXPECT_FALSE(ragged.logicallyEquals(otherLayout));
    EXPECT_FALSE(ragged == otherLayout);

    // Logically equal offset tensors that differ only in name and UID.
    auto renamedOffset = makeEqualityTensor(1, "t");
    renamedOffset.set_ragged_offset(makeRaggedOffset(4, 11, "offs_other"));
    EXPECT_TRUE(ragged.logicallyEquals(renamedOffset));
    EXPECT_FALSE(ragged == renamedOffset);

    // Strictly equal offset tensors held by distinct objects.
    auto sameOffset = makeEqualityTensor(1, "t");
    sameOffset.set_ragged_offset(makeRaggedOffset(4, 10, "offs"));
    EXPECT_TRUE(ragged.logicallyEquals(sameOffset));
    EXPECT_TRUE(ragged == sameOffset);
}

TEST(TestTensorAttributes, LogicalEqualityDistinguishesRaggedOffsetMultiplier)
{
    const auto offset = makeRaggedOffset(4, 10, "offs");
    auto tensorA = makeEqualityTensor(1, "t");
    tensorA.set_ragged_offset(offset);
    auto tensorB = makeEqualityTensor(1, "t");
    tensorB.set_ragged_offset(offset);
    ASSERT_TRUE(tensorA == tensorB);

    tensorB.set_ragged_offset_multiplier(2);
    EXPECT_FALSE(tensorA.logicallyEquals(tensorB));
    EXPECT_FALSE(tensorA == tensorB);
}
