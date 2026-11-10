// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSoftmax.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

namespace
{

const float K_NEG_INF = -std::numeric_limits<float>::infinity();

template <typename Type>
Tensor<Type> makeSoftmaxTensor(const std::vector<int64_t>& dims)
{
    return Tensor<Type>(dims, generateStrides(dims));
}

// Writes `values` in logical row-major order through the tensor's own strides.
template <typename Type>
void setLogicalValues(Tensor<Type>& tensor, const std::vector<float>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    const auto& dims = tensor.dims();
    std::vector<int64_t> index(dims.size(), 0);
    for(const auto value : values)
    {
        tensor.setHostValue(static_cast<Type>(value), index);
        for(size_t d = dims.size(); d-- > 0;)
        {
            if(++index[d] < dims[d])
            {
                break;
            }
            index[d] = 0;
        }
    }
}

// Reads the tensor in logical row-major order through its own strides.
template <typename Type>
std::vector<float> logicalValues(const Tensor<Type>& tensor)
{
    std::vector<float> result;
    const auto& dims = tensor.dims();
    const auto count = tensor.elementCount();
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t i = 0; i < count; ++i)
    {
        result.push_back(static_cast<float>(tensor.getHostValue(index)));
        for(size_t d = dims.size(); d-- > 0;)
        {
            if(++index[d] < dims[d])
            {
                break;
            }
            index[d] = 0;
        }
    }
    return result;
}

void expectNear(const std::vector<float>& actual,
                const std::vector<float>& expected,
                float tolerance = 1e-6f)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "at index " << i;
    }
}

TensorBase<float>* noStats()
{
    return nullptr;
}

} // namespace

template <typename T>
class CpuFpReferenceSoftmaxTyped : public ::testing::Test
{
};

using SoftmaxDataTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(CpuFpReferenceSoftmaxTyped, SoftmaxDataTypes, );

// Row [0, ln 3]: exp(p - max) = [1/3, 1], sum = 4/3, s = [1/4, 3/4] (exact in every type);
// stats = ln 3 + ln(4/3) = ln 4.
TYPED_TEST(CpuFpReferenceSoftmaxTyped, TwoElementRow)
{
    auto p = makeSoftmaxTensor<TypeParam>({1, 2});
    setLogicalValues(p, {0.0f, std::log(3.0f)});
    auto s = makeSoftmaxTensor<TypeParam>({1, 2});
    auto stats = makeSoftmaxTensor<float>({1, 1});

    CpuFpReferenceSoftmax::forward(p, s, &stats, false);

    expectNear(logicalValues(s), {0.25f, 0.75f}, 1e-2f);
    expectNear(logicalValues(stats), {std::log(4.0f)}, 1e-2f);
}

// Row 0 = [0, ln 2, ln 3]: exp(p) = [1, 2, 3], s = [1, 2, 3] / 6, stats = ln 6.
// Row 1 = [1, 1, 1]: s = 1/3 each, stats = 1 + ln 3.
TEST(TestCpuFpReferenceSoftmax, RowsAreIndependent)
{
    auto p = makeSoftmaxTensor<float>({2, 3});
    setLogicalValues(p, {0.0f, std::log(2.0f), std::log(3.0f), 1.0f, 1.0f, 1.0f});
    auto s = makeSoftmaxTensor<float>({2, 3});
    auto stats = makeSoftmaxTensor<float>({2, 1});

    CpuFpReferenceSoftmax::forward(p, s, &stats, false);

    expectNear(logicalValues(s),
               {1.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f});
    expectNear(logicalValues(stats), {std::log(6.0f), 1.0f + std::log(3.0f)}, 1e-5f);
}

// stats_use_log2 divides the log-sum-exp by ln 2: row [0, 0, 0, 0] -> ln 4 / ln 2 = 2.
TEST(TestCpuFpReferenceSoftmax, StatsUseLog2)
{
    auto p = makeSoftmaxTensor<float>({1, 4});
    setLogicalValues(p, {0.0f, 0.0f, 0.0f, 0.0f});
    auto s = makeSoftmaxTensor<float>({1, 4});
    auto stats = makeSoftmaxTensor<float>({1, 1});

    CpuFpReferenceSoftmax::forward(p, s, &stats, true);

    expectNear(logicalValues(s), {0.25f, 0.25f, 0.25f, 0.25f});
    expectNear(logicalValues(stats), {2.0f}, 1e-6f);
}

// Rank 1 is a single row. Stats is optional.
TEST(TestCpuFpReferenceSoftmax, RankOneWithoutStats)
{
    auto p = makeSoftmaxTensor<float>({2});
    setLogicalValues(p, {3.0f, 3.0f});
    auto s = makeSoftmaxTensor<float>({2});

    CpuFpReferenceSoftmax::forward(p, s, noStats(), false);

    expectNear(logicalValues(s), {0.5f, 0.5f});
}

// Max subtraction keeps large logits finite: [1000, 1000] -> 1/2 each, stats 1000 + ln 2.
TEST(TestCpuFpReferenceSoftmax, LargeLogitsAreStable)
{
    auto p = makeSoftmaxTensor<float>({1, 2});
    setLogicalValues(p, {1000.0f, 1000.0f});
    auto s = makeSoftmaxTensor<float>({1, 2});
    auto stats = makeSoftmaxTensor<float>({1, 1});

    CpuFpReferenceSoftmax::forward(p, s, &stats, false);

    expectNear(logicalValues(s), {0.5f, 0.5f});
    expectNear(logicalValues(stats), {1000.0f + std::log(2.0f)}, 1e-3f);
}

// A partly masked row ignores the -inf entries; a fully masked row has s = 0 and
// stats = -inf (no NaN).
TEST(TestCpuFpReferenceSoftmax, MaskedRows)
{
    auto p = makeSoftmaxTensor<float>({2, 2});
    setLogicalValues(p, {K_NEG_INF, 0.0f, K_NEG_INF, K_NEG_INF});
    auto s = makeSoftmaxTensor<float>({2, 2});
    auto stats = makeSoftmaxTensor<float>({2, 1});

    CpuFpReferenceSoftmax::forward(p, s, &stats, false);

    expectNear(logicalValues(s), {0.0f, 1.0f, 0.0f, 0.0f});
    const auto statsValues = logicalValues(stats);
    EXPECT_FLOAT_EQ(statsValues[0], 0.0f);
    EXPECT_EQ(statsValues[1], K_NEG_INF);
}

// A NaN logit poisons its row only.
TEST(TestCpuFpReferenceSoftmax, NanPropagatesWithinRow)
{
    auto p = makeSoftmaxTensor<float>({2, 2});
    setLogicalValues(p, {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f});
    auto s = makeSoftmaxTensor<float>({2, 2});
    auto stats = makeSoftmaxTensor<float>({2, 1});

    CpuFpReferenceSoftmax::forward(p, s, &stats, false);

    const auto sValues = logicalValues(s);
    EXPECT_TRUE(std::isnan(sValues[0]));
    EXPECT_TRUE(std::isnan(sValues[1]));
    EXPECT_FLOAT_EQ(sValues[2], 0.5f);
    EXPECT_FLOAT_EQ(sValues[3], 0.5f);
    const auto statsValues = logicalValues(stats);
    EXPECT_TRUE(std::isnan(statsValues[0]));
    EXPECT_NEAR(statsValues[1], std::log(2.0f), 1e-6f);
}

// Strided (column-major) tensors: the softmax still runs over the logical last dim.
TEST(TestCpuFpReferenceSoftmax, NonContiguousStrides)
{
    Tensor<float> p(std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 2});
    setLogicalValues(p, {0.0f, std::log(3.0f), 5.0f, 5.0f});
    Tensor<float> s(std::vector<int64_t>{2, 2}, std::vector<int64_t>{1, 2});
    auto stats = makeSoftmaxTensor<float>({2, 1});

    CpuFpReferenceSoftmax::forward(p, s, &stats, false);

    expectNear(logicalValues(s), {0.25f, 0.75f, 0.5f, 0.5f});
    expectNear(logicalValues(stats), {std::log(4.0f), 5.0f + std::log(2.0f)}, 1e-5f);
}

// s may alias p: each element is read before its output is written.
TEST(TestCpuFpReferenceSoftmax, InPlace)
{
    auto p = makeSoftmaxTensor<float>({1, 3});
    setLogicalValues(p, {0.0f, std::log(2.0f), std::log(3.0f)});

    CpuFpReferenceSoftmax::forward(p, p, noStats(), false);

    expectNear(logicalValues(p), {1.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f});
}

// HALF logits may produce FLOAT probabilities.
TEST(TestCpuFpReferenceSoftmax, HalfInFloatOut)
{
    auto p = makeSoftmaxTensor<half>({1, 2});
    setLogicalValues(p, {1.0f, 1.0f});
    auto s = makeSoftmaxTensor<float>({1, 2});

    CpuFpReferenceSoftmax::forward(p, s, noStats(), false);

    expectNear(logicalValues(s), {0.5f, 0.5f});
}

TEST(TestCpuFpReferenceSoftmax, RejectsMismatchedShapes)
{
    auto p = makeSoftmaxTensor<float>({2, 3});
    setLogicalValues(p, {0, 0, 0, 0, 0, 0});
    auto badS = makeSoftmaxTensor<float>({2, 2});
    EXPECT_THROW(CpuFpReferenceSoftmax::forward(p, badS, noStats(), false), std::runtime_error);

    auto s = makeSoftmaxTensor<float>({2, 3});
    auto badStats = makeSoftmaxTensor<float>({2, 3});
    EXPECT_THROW(CpuFpReferenceSoftmax::forward(p, s, &badStats, false), std::runtime_error);
}
