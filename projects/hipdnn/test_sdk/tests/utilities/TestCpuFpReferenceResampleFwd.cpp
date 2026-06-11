// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_common_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleFwd.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_test_sdk::utilities;
using hipdnn_test_sdk::detail::safeTestTypeCast;

namespace
{

template <typename Type>
void fillValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = safeTestTypeCast<Type>(values[i]);
    }
    tensor.memory().markHostModified();
}

template <typename Type>
void expectTensorValues(const Tensor<Type>& tensor, const std::vector<float>& expected)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), expected.size());
    const auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(static_cast<float>(data[i]), expected[i]) << "Mismatch at index " << i;
    }
}

} // namespace

TEST(TestCpuFpReferenceResampleFwd, MaxPoolProducesValuesAndIndices)
{
    Tensor<float> x({1, 1, 3, 3});
    Tensor<float> y({1, 1, 2, 2});
    Tensor<int32_t> index({1, 1, 2, 2});

    fillValues(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

    CpuFpReferenceResampleFwd::forward<float, float, float, int32_t>(
        x, y, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD, &index);

    expectTensorValues(y, {5.0f, 6.0f, 8.0f, 9.0f});
    expectTensorValues(index, {4.0f, 5.0f, 7.0f, 8.0f});
}

TEST(TestCpuFpReferenceResampleFwd, AverageExcludePaddingUsesValidElementCount)
{
    Tensor<float> x({1, 1, 2, 2});
    Tensor<float> y({1, 1, 2, 2});

    fillValues(x, {1.0f, 2.0f, 3.0f, 4.0f});

    CpuFpReferenceResampleFwd::forward<float, float, float>(
        x, y, {1, 1}, {1, 1}, {2, 2}, ResampleMode::AVGPOOL_EXCLUDE_PADDING, PaddingMode::ZERO_PAD);

    expectTensorValues(y, {1.0f, 1.5f, 2.0f, 2.5f});
}

TEST(TestCpuFpReferenceResampleFwd, SupportsChannelLastStrides)
{
    Tensor<float> x({1, 2, 3, 3}, TensorLayout::NHWC);
    Tensor<float> y({1, 2, 2, 2}, TensorLayout::NHWC);

    fillValues(x,
               {1.0f,
                10.0f,
                2.0f,
                20.0f,
                3.0f,
                30.0f,
                4.0f,
                40.0f,
                5.0f,
                50.0f,
                6.0f,
                60.0f,
                7.0f,
                70.0f,
                8.0f,
                80.0f,
                9.0f,
                90.0f});

    CpuFpReferenceResampleFwd::forward<float, float, float>(
        x, y, {0, 0}, {1, 1}, {2, 2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD);

    expectTensorValues(y, {5.0f, 50.0f, 6.0f, 60.0f, 8.0f, 80.0f, 9.0f, 90.0f});
}

TEST(TestCpuFpReferenceResampleFwd, RejectsUnsupportedRanks)
{
    const Tensor<float> x({1, 2, 3});
    Tensor<float> y({1, 2, 2});

    EXPECT_THROW((CpuFpReferenceResampleFwd::forward<float, float, float>(
                     x, y, {0}, {1}, {2}, ResampleMode::MAXPOOL, PaddingMode::ZERO_PAD)),
                 std::runtime_error);
}
