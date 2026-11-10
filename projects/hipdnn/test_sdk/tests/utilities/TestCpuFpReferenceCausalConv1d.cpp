// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceCausalConv1d.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using Activation = CpuFpReferenceCausalConv1d::Activation;

namespace
{

/// A packed tensor holding `values` in memory order.
template <typename Type>
Tensor<Type> makeConvTensor(const std::vector<int64_t>& dims, const std::vector<float>& values)
{
    Tensor<Type> tensor(dims, generateStrides(dims));
    EXPECT_EQ(tensor.elementSpace(), values.size());
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size() && i < tensor.elementSpace(); ++i)
    {
        data[i] = static_cast<Type>(values[i]);
    }
    return tensor;
}

template <typename IndexType>
Tensor<IndexType> createIndices(const std::vector<int64_t>& values)
{
    const std::vector<int64_t> dims = {static_cast<int64_t>(values.size())};
    Tensor<IndexType> tensor(dims, generateStrides(dims));
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = static_cast<IndexType>(values[i]);
    }
    return tensor;
}

template <typename Type>
void expectValues(const TensorBase<Type>& tensor, const std::vector<float>& expected)
{
    ASSERT_EQ(tensor.elementCount(), expected.size());
    const auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(static_cast<float>(data[i]), expected[i]) << "Mismatch at flat index " << i;
    }
}

} // namespace

// One sequence without a state equals a convolution over x zero-padded by W - 1 on the left.
// C = 2, W = 3, x[:, 0] = [1, 2, 3, 4], x[:, 1] = [-1, 0, 1, 2],
// weight[0] = [1, 2, 3], weight[1] = [0.5, 0, -1]:
//   channel 0: [3, 2 + 6, 1 + 4 + 9, 2 + 6 + 12] = [3, 8, 14, 20]
//   channel 1: [1, 0, -0.5 - 1, -2]
// The final state is the last two padded inputs of each channel: [3, 4] and [1, 2].
TEST(TestCpuFpReferenceCausalConv1d, SingleSequenceWithoutStateIsZeroPaddedConvolution)
{
    auto x = makeConvTensor<float>({4, 2}, {1, -1, 2, 0, 3, 1, 4, 2});
    auto weight = makeConvTensor<float>({2, 3}, {1, 2, 3, 0.5F, 0, -1});
    auto cuSeqlens = createIndices<int32_t>({0, 4});
    auto y = makeConvTensor<float>({4, 2}, std::vector<float>(8, 0.0F));
    auto finalState = makeConvTensor<float>({1, 2, 2}, std::vector<float>(4, 0.0F));

    CpuFpReferenceCausalConv1d::forward<float, int32_t>(
        x, weight, cuSeqlens, y, Activation::IDENTITY, nullptr, nullptr, nullptr, &finalState);

    expectValues(y, {3, 1, 8, 0, 14, -1.5F, 20, -2});
    expectValues(finalState, {3, 4, 1, 2});
}

template <typename T>
class CpuFpReferenceCausalConv1dTyped : public ::testing::Test
{
};

using CausalConv1dDataTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(CpuFpReferenceCausalConv1dTyped, CausalConv1dDataTypes, );

// Three packed sequences, C = 1, W = 4, cu_seqlens [0, 0, 1, 4], weight [1, 2, 3, 4]:
//   sequence 0 is empty: no output, its final state is its initial state [10, 20, 30];
//   sequence 1 has one token (< W - 1): xs [40, 50, 60, 1] -> y 324, final [50, 60, 1]
//     (the tail of the initial state followed by the token);
//   sequence 2: xs [7, 8, 9, 2, 3, 4] -> y [58, 44, 38], final [2, 3, 4].
TYPED_TEST(CpuFpReferenceCausalConv1dTyped, PackedSequencesWithEmptyAndShortSequences)
{
    auto x = makeConvTensor<TypeParam>({4, 1}, {1, 2, 3, 4});
    auto weight = makeConvTensor<TypeParam>({1, 4}, {1, 2, 3, 4});
    auto cuSeqlens = createIndices<int32_t>({0, 0, 1, 4});
    auto y = makeConvTensor<TypeParam>({4, 1}, {0, 0, 0, 0});
    auto initialState = makeConvTensor<TypeParam>({3, 1, 3}, {10, 20, 30, 40, 50, 60, 7, 8, 9});
    auto finalState = makeConvTensor<TypeParam>({3, 1, 3}, std::vector<float>(9, 0.0F));

    CpuFpReferenceCausalConv1d::forward<TypeParam, int32_t>(x,
                                                            weight,
                                                            cuSeqlens,
                                                            y,
                                                            Activation::IDENTITY,
                                                            nullptr,
                                                            &initialState,
                                                            nullptr,
                                                            &finalState);

    expectValues(y, {324, 58, 44, 38});
    expectValues(finalState, {10, 20, 30, 50, 60, 1, 2, 3, 4});
}

// Decoding: T = N with one token per sequence and INT64 cu_seqlens [0, 1, 2]. C = 1, W = 3,
// weight [1, 10, 100]. Each state rolls left by one and takes the new token:
//   sequence 0: xs [1, 2, 5] -> y 521, final [2, 5]
//   sequence 1: xs [3, 4, 6] -> y 643, final [4, 6]
TEST(TestCpuFpReferenceCausalConv1d, DecodeOneTokenPerSequenceRollsState)
{
    auto x = makeConvTensor<float>({2, 1}, {5, 6});
    auto weight = makeConvTensor<float>({1, 3}, {1, 10, 100});
    auto cuSeqlens = createIndices<int64_t>({0, 1, 2});
    auto y = makeConvTensor<float>({2, 1}, {0, 0});
    auto initialState = makeConvTensor<float>({2, 1, 2}, {1, 2, 3, 4});
    auto finalState = makeConvTensor<float>({2, 1, 2}, {0, 0, 0, 0});

    CpuFpReferenceCausalConv1d::forward<float, int64_t>(x,
                                                        weight,
                                                        cuSeqlens,
                                                        y,
                                                        Activation::IDENTITY,
                                                        nullptr,
                                                        &initialState,
                                                        nullptr,
                                                        &finalState);

    expectValues(y, {521, 643});
    expectValues(finalState, {2, 5, 4, 6});
}

// state_indices [2, 0] address a pool of P = 3 rows; C = 1, W = 2, weight [1, 1]:
//   sequence 0 reads row 2 ([3]): xs [3, 10] -> y 13, final row 2 = [10]
//   sequence 1 reads row 0 ([1]): xs [1, 20] -> y 21, final row 0 = [20]
// Row 1 of the separate final pool is not addressed and keeps its -1 prefill.
TEST(TestCpuFpReferenceCausalConv1d, StateIndicesAddressPool)
{
    auto x = makeConvTensor<float>({2, 1}, {10, 20});
    auto weight = makeConvTensor<float>({1, 2}, {1, 1});
    auto cuSeqlens = createIndices<int32_t>({0, 1, 2});
    auto y = makeConvTensor<float>({2, 1}, {0, 0});
    auto initialState = makeConvTensor<float>({3, 1, 1}, {1, 2, 3});
    auto stateIndices = createIndices<int64_t>({2, 0});
    auto finalState = makeConvTensor<float>({3, 1, 1}, {-1, -1, -1});

    CpuFpReferenceCausalConv1d::forward<float, int32_t, int64_t>(x,
                                                                 weight,
                                                                 cuSeqlens,
                                                                 y,
                                                                 Activation::IDENTITY,
                                                                 nullptr,
                                                                 &initialState,
                                                                 &stateIndices,
                                                                 &finalState);

    expectValues(y, {13, 21});
    expectValues(finalState, {20, -1, 10});
}

// final_state bound to initial_state's buffer (two views of one allocation). C = 1, W = 3,
// weight [1, 1, 1], cu_seqlens [0, 2, 3], state [[1, 2], [3, 4]]:
//   sequence 0: xs [1, 2, 5, 6] -> y [8, 13], state 0 becomes [5, 6]
//   sequence 1: xs [3, 4, 7]    -> y 14,      state 1 becomes [4, 7]
TEST(TestCpuFpReferenceCausalConv1d, InPlaceStateUpdate)
{
    auto x = makeConvTensor<float>({3, 1}, {5, 6, 7});
    auto weight = makeConvTensor<float>({1, 3}, {1, 1, 1});
    auto cuSeqlens = createIndices<int32_t>({0, 2, 3});
    auto y = makeConvTensor<float>({3, 1}, {0, 0, 0});
    auto state = makeConvTensor<float>({2, 1, 2}, {1, 2, 3, 4});
    ShallowTensor<float> initialView(state.memory().hostData(), state.dims(), state.strides());
    ShallowTensor<float> finalView(state.memory().hostData(), state.dims(), state.strides());

    CpuFpReferenceCausalConv1d::forward<float, int32_t>(
        x, weight, cuSeqlens, y, Activation::IDENTITY, nullptr, &initialView, nullptr, &finalView);

    expectValues(y, {8, 13, 14});
    expectValues(state, {5, 6, 4, 7});
}

// In-place update of a state pool through permuted state_indices [1, 0]; final_state is
// bound to initial_state's buffer, and each sequence reads and writes its own row.
// C = 1, W = 2, weight [1, 1], pool [[1], [2]], x [10, 20]:
//   sequence 0 reads row 1 ([2]) -> y 12, row 1 = [10]
//   sequence 1 reads row 0 ([1]) -> y 21, row 0 = [20]
TEST(TestCpuFpReferenceCausalConv1d, InPlaceStatePoolUpdate)
{
    auto x = makeConvTensor<float>({2, 1}, {10, 20});
    auto weight = makeConvTensor<float>({1, 2}, {1, 1});
    auto cuSeqlens = createIndices<int32_t>({0, 1, 2});
    auto y = makeConvTensor<float>({2, 1}, {0, 0});
    auto pool = makeConvTensor<float>({2, 1, 1}, {1, 2});
    auto stateIndices = createIndices<int32_t>({1, 0});
    ShallowTensor<float> initialView(pool.memory().hostData(), pool.dims(), pool.strides());
    ShallowTensor<float> finalView(pool.memory().hostData(), pool.dims(), pool.strides());

    CpuFpReferenceCausalConv1d::forward<float, int32_t, int32_t>(x,
                                                                 weight,
                                                                 cuSeqlens,
                                                                 y,
                                                                 Activation::IDENTITY,
                                                                 nullptr,
                                                                 &initialView,
                                                                 &stateIndices,
                                                                 &finalView);

    expectValues(y, {12, 21});
    expectValues(pool, {20, 10});
}

// Bias and SILU: C = 1, W = 2, weight [1, 1], bias [0.5], x [1, -2]:
//   pre-activation [0.5 + 0 + 1, 0.5 + 1 - 2] = [1.5, -0.5], then v * sigmoid(v).
TEST(TestCpuFpReferenceCausalConv1d, BiasAndSilu)
{
    auto x = makeConvTensor<float>({2, 1}, {1, -2});
    auto weight = makeConvTensor<float>({1, 2}, {1, 1});
    auto bias = makeConvTensor<float>({1}, {0.5F});
    auto cuSeqlens = createIndices<int32_t>({0, 2});
    auto y = makeConvTensor<float>({2, 1}, {0, 0});

    CpuFpReferenceCausalConv1d::forward<float, int32_t>(
        x, weight, cuSeqlens, y, Activation::SILU, &bias);

    const auto silu = [](float v) { return v / (1.0F + std::exp(-v)); };
    const auto* data = y.memory().hostData();
    EXPECT_NEAR(data[0], silu(1.5F), 1e-6F);
    EXPECT_NEAR(data[1], silu(-0.5F), 1e-6F);
}

// Bias alone, IDENTITY: the bias is added once per output.
TEST(TestCpuFpReferenceCausalConv1d, BiasWithIdentity)
{
    auto x = makeConvTensor<float>({2, 1}, {1, -2});
    auto weight = makeConvTensor<float>({1, 2}, {1, 1});
    auto bias = makeConvTensor<float>({1}, {0.5F});
    auto cuSeqlens = createIndices<int32_t>({0, 2});
    auto y = makeConvTensor<float>({2, 1}, {0, 0});

    CpuFpReferenceCausalConv1d::forward<float, int32_t>(
        x, weight, cuSeqlens, y, Activation::IDENTITY, &bias);

    expectValues(y, {1.5F, -0.5F});
}

TEST(TestCpuFpReferenceCausalConv1d, RejectsStateIndicesWithoutInitialState)
{
    auto x = makeConvTensor<float>({2, 1}, {1, 2});
    auto weight = makeConvTensor<float>({1, 2}, {1, 1});
    auto cuSeqlens = createIndices<int32_t>({0, 1, 2});
    auto y = makeConvTensor<float>({2, 1}, {0, 0});
    auto stateIndices = createIndices<int32_t>({0, 1});

    EXPECT_THROW(
        (CpuFpReferenceCausalConv1d::forward<float, int32_t, int32_t>(
            x, weight, cuSeqlens, y, Activation::IDENTITY, nullptr, nullptr, &stateIndices)),
        std::invalid_argument);
}

TEST(TestCpuFpReferenceCausalConv1d, RejectsStateIndexOutsidePool)
{
    auto x = makeConvTensor<float>({2, 1}, {1, 2});
    auto weight = makeConvTensor<float>({1, 2}, {1, 1});
    auto cuSeqlens = createIndices<int32_t>({0, 1, 2});
    auto y = makeConvTensor<float>({2, 1}, {0, 0});
    auto pool = makeConvTensor<float>({2, 1, 1}, {0, 0});
    auto stateIndices = createIndices<int32_t>({0, 2});

    EXPECT_THROW((CpuFpReferenceCausalConv1d::forward<float, int32_t, int32_t>(
                     x, weight, cuSeqlens, y, Activation::IDENTITY, nullptr, &pool, &stateIndices)),
                 std::out_of_range);
}

TEST(TestCpuFpReferenceCausalConv1d, RejectsTwoSequencesWritingOneStateRow)
{
    auto x = makeConvTensor<float>({2, 1}, {1, 2});
    auto weight = makeConvTensor<float>({1, 2}, {1, 1});
    auto cuSeqlens = createIndices<int32_t>({0, 1, 2});
    auto y = makeConvTensor<float>({2, 1}, {0, 0});
    auto pool = makeConvTensor<float>({2, 1, 1}, {0, 0});
    auto finalPool = makeConvTensor<float>({2, 1, 1}, {0, 0});
    auto stateIndices = createIndices<int32_t>({1, 1});

    EXPECT_THROW((CpuFpReferenceCausalConv1d::forward<float, int32_t, int32_t>(x,
                                                                               weight,
                                                                               cuSeqlens,
                                                                               y,
                                                                               Activation::IDENTITY,
                                                                               nullptr,
                                                                               &pool,
                                                                               &stateIndices,
                                                                               &finalPool)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceCausalConv1d, RejectsOffsetsNotCoveringTokens)
{
    auto x = makeConvTensor<float>({3, 1}, {1, 2, 3});
    auto weight = makeConvTensor<float>({1, 2}, {1, 1});
    auto y = makeConvTensor<float>({3, 1}, {0, 0, 0});

    auto shortOffsets = createIndices<int32_t>({0, 2});
    EXPECT_THROW((CpuFpReferenceCausalConv1d::forward<float, int32_t>(x, weight, shortOffsets, y)),
                 std::invalid_argument);

    auto decreasingOffsets = createIndices<int32_t>({0, 2, 1, 3});
    EXPECT_THROW(
        (CpuFpReferenceCausalConv1d::forward<float, int32_t>(x, weight, decreasingOffsets, y)),
        std::invalid_argument);
}

TEST(TestCpuFpReferenceCausalConv1d, RejectsBadShapes)
{
    auto x = makeConvTensor<float>({2, 2}, {1, 2, 3, 4});
    auto cuSeqlens = createIndices<int32_t>({0, 2});
    auto y = makeConvTensor<float>({2, 2}, {0, 0, 0, 0});

    auto narrowWeight = makeConvTensor<float>({2, 1}, {1, 1});
    EXPECT_THROW(
        (CpuFpReferenceCausalConv1d::forward<float, int32_t>(x, narrowWeight, cuSeqlens, y)),
        std::invalid_argument);

    auto weight = makeConvTensor<float>({2, 3}, {1, 1, 1, 1, 1, 1});
    auto wrongState = makeConvTensor<float>({1, 2, 3}, std::vector<float>(6, 0.0F));
    EXPECT_THROW((CpuFpReferenceCausalConv1d::forward<float, int32_t>(
                     x, weight, cuSeqlens, y, Activation::IDENTITY, nullptr, &wrongState)),
                 std::invalid_argument);

    auto wrongBias = makeConvTensor<float>({3}, {0, 0, 0});
    EXPECT_THROW((CpuFpReferenceCausalConv1d::forward<float, int32_t>(
                     x, weight, cuSeqlens, y, Activation::IDENTITY, &wrongBias)),
                 std::invalid_argument);
}
